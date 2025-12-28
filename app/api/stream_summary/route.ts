import { streamText } from "ai"
import { ThreadProcessor } from "@/lib/backend/thread-processor"
import { PromptTemplates } from "@/lib/backend/prompts"
import { HighlightRanker, type ClassifiedSentence } from "@/lib/backend/highlight-ranker"
import { ProcessingMetrics } from "@/lib/backend/metrics"
import type { HighlightCategory, Persona } from "@/lib/types"

const processor = new ThreadProcessor()
const prompts = new PromptTemplates()
const ranker = new HighlightRanker()

export async function POST(request: Request) {
  const metrics = new ProcessingMetrics()
  metrics.start()

  try {
    const body = await request.json()
    const { text, persona = "developer", thread, max_highlights = 10 } = body

    let threadText = text
    let processedThread = null

    // Process thread if full object provided
    if (thread && thread.posts) {
      processedThread = processor.process(thread)
      threadText = processedThread.text
    }

    if (!threadText) {
      return new Response(JSON.stringify({ error: "No text provided" }), { status: 400 })
    }

    // Create streaming response
    const encoder = new TextEncoder()
    let streamedSummary = ""
    let totalTokens = 0

    const stream = new ReadableStream({
      async start(controller) {
        try {
          console.log("[v0] Starting two-pass summarization")

          // === PASS A: Fast streaming draft (low latency) ===
          const streamingPrompt = prompts.buildStreamingSummaryPrompt(threadText, persona as Persona)

          const { textStream, usage } = await streamText({
            model: "openai/gpt-4o-mini", // Fast model for streaming
            prompt: streamingPrompt,
            temperature: 0.2,
            maxTokens: 400,
          })

          // Stream tokens to client
          for await (const token of textStream) {
            if (!metrics.getMetrics().firstTokenTime) {
              metrics.recordFirstToken()
            }

            streamedSummary += token
            totalTokens++

            const event = {
              type: "token",
              data: { token, totalTokens },
              timestamp: new Date().toISOString(),
            }
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
          }

          console.log("[v0] Pass A complete, streamed summary:", streamedSummary.slice(0, 100))

          // Record streaming metrics
          if (usage) {
            metrics.recordTokens(usage.totalTokens, "gpt-4o-mini")
          }

          // === PASS B: Quality consolidation (parallel) ===
          const consolidationPrompt = prompts.buildConsolidationPrompt(threadText, persona as Persona, streamedSummary)

          const consolidation = generateText({
            model: "openai/gpt-4o-mini",
            prompt: consolidationPrompt,
            temperature: 0.1,
            maxTokens: 500,
          })

          // === Generate digest (parallel) ===
          const digestPrompt = prompts.buildDigestPrompt(threadText, persona as Persona)

          const digestGen = generateText({
            model: "openai/gpt-4o-mini",
            prompt: digestPrompt,
            temperature: 0.2,
            maxTokens: 300,
          })

          // Wait for both
          const [consolidatedResult, digestResult] = await Promise.all([consolidation, digestGen])

          const finalSummary = consolidatedResult.text
          const digestText = digestResult.text

          console.log("[v0] Pass B complete, refined summary")

          // Parse digest bullets
          const bullets = digestText
            .split("\n")
            .filter((line) => line.trim().startsWith("-"))
            .map((line) => line.replace(/^-\s*/, "").trim())
            .slice(0, 5)

          // Extract verdict
          const verdictMatch = digestText.match(/Verdict:\s*(.+)/i)
          const verdict = verdictMatch ? verdictMatch[1].trim() : null

          if (verdict) bullets.push(`Verdict: ${verdict}`)

          metrics.recordDigest()

          // === Extract and classify highlights ===
          let highlights: any[] = []

          if (processedThread && processedThread.sentences.length > 0) {
            // Use LLM to extract highlights with provenance
            const highlightPrompt = prompts.buildHighlightExtractionPrompt(
              processedThread.sentences.map((s) => s.text),
              max_highlights,
            )

            const { text: highlightResponse } = await generateText({
              model: "openai/gpt-4o-mini",
              prompt: highlightPrompt,
              temperature: 0.1,
              maxTokens: 1500,
            })

            try {
              const highlightData = JSON.parse(highlightResponse)

              // Map to full sentences with metadata
              const classifiedSentences: ClassifiedSentence[] = highlightData.map((h: any) => {
                const sentence = processedThread.sentences[h.sentence_index]
                return {
                  ...sentence,
                  category: h.category as HighlightCategory,
                  confidence: h.confidence,
                  why_matters: h.why_matters,
                  importance_score: 0,
                }
              })

              // Rank and select
              const scored = ranker.calculateImportanceScores(classifiedSentences, persona as Persona, finalSummary)
              const topHighlights = ranker.selectTopHighlights(scored, max_highlights)

              highlights = topHighlights.map((h, i) => ({
                id: `hl_${i}_${Date.now()}`,
                text: h.text,
                category: h.category,
                confidence: h.confidence,
                importance_score: h.importance_score,
                why_matters: h.why_matters,
                post_id: h.post_id,
                author_id: h.author_id,
                post_position: h.post_position,
                sentence_index: h.sentence_index,
                start_offset: h.char_offset,
                end_offset: h.char_offset + h.text.length,
              }))

              const avgConfidence = highlights.reduce((sum, h) => sum + h.confidence, 0) / highlights.length
              metrics.recordHighlights(highlights.length, avgConfidence)
            } catch (e) {
              console.error("[v0] Failed to parse highlight response:", e)
            }
          }

          // Send consolidated summary update
          const summaryUpdateEvent = {
            type: "summary_update",
            data: {
              final_summary: finalSummary,
              confidence: "high",
            },
            timestamp: new Date().toISOString(),
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(summaryUpdateEvent)}\n\n`))

          // Send digest
          const digestEvent = {
            type: "digest",
            data: { bullets, highlights },
            timestamp: new Date().toISOString(),
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(digestEvent)}\n\n`))

          // Stream individual highlights
          for (const highlight of highlights) {
            const highlightEvent = {
              type: "highlight",
              data: { highlight },
              timestamp: new Date().toISOString(),
            }
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(highlightEvent)}\n\n`))
            await new Promise((resolve) => setTimeout(resolve, 100))
          }

          // Calculate metrics
          metrics.finish()
          const metricsSummary = metrics.getSummary()

          console.log("[v0] Metrics:", metricsSummary)

          // Send completion with metrics
          const completeEvent = {
            type: "complete",
            data: {
              message: "Summary complete",
              metrics: metricsSummary,
              read_time_saved: processedThread
                ? Math.max(0, processor.estimateReadTime(threadText) - processor.estimateReadTime(finalSummary))
                : 0,
            },
            timestamp: new Date().toISOString(),
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(completeEvent)}\n\n`))

          controller.close()
        } catch (error: any) {
          console.error("[v0] Stream error:", error)
          const errorEvent = {
            type: "error",
            data: {
              error: error.message,
              phase: "streaming",
            },
            timestamp: new Date().toISOString(),
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(errorEvent)}\n\n`))
          controller.close()
        }
      },
    })

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    })
  } catch (error: any) {
    console.error("[v0] API error:", error)
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    })
  }
}

// Import generateText at top
import { generateText } from "ai"
