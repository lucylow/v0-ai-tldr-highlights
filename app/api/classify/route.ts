import { NextResponse } from "next/server"
import { generateObject } from "ai"
import { z } from "zod"

const classificationSchema = z.object({
  classifications: z.array(
    z.object({
      sentence_index: z.number(),
      category: z.enum(["fact", "solution", "opinion", "question", "citation", "irrelevant"]),
      confidence: z.number().min(0).max(1),
      reasoning: z.string().optional(),
    }),
  ),
})

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { sentences } = body

    if (!sentences || !Array.isArray(sentences)) {
      return NextResponse.json({ error: "Sentences array required" }, { status: 400 })
    }

    const { object } = await generateObject({
      model: "openai/gpt-4o-mini",
      schema: classificationSchema,
      prompt: `You are a sentence classifier for forum threads. Classify each sentence into one of these categories:

- fact: Objective information, data, or verifiable statements
- solution: Proposed solutions, fixes, or answers to problems
- opinion: Personal views, subjective statements, or preferences
- question: Questions or requests for information
- citation: References, links, or quoted sources
- irrelevant: Off-topic, spam, or non-informative content

Sentences to classify:
${sentences.map((s: string, i: number) => `${i}. ${s}`).join("\n")}

Provide a confidence score (0-1) for each classification.`,
      temperature: 0.1,
    })

    return NextResponse.json({ classifications: object.classifications })
  } catch (error: any) {
    console.error("[API] Classify error:", error)
    return NextResponse.json({ error: error.message || "Failed to classify sentences" }, { status: 500 })
  }
}
