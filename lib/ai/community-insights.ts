// AI-powered community insights leveraging Foru.ms data
import { generateObject } from "ai"
import { z } from "zod"
import type { ForumsThread, ForumsPost } from "@/lib/forums/client"

export interface CommunityInsight {
  type: "sentiment" | "topic" | "expertise" | "trend" | "suggestion"
  title: string
  description: string
  confidence: number
  metadata?: Record<string, any>
}

/**
 * Analyze sentiment across a thread
 * Showcases: Easy LLM integration with Foru.ms data
 */
export async function analyzeSentiment(posts: ForumsPost[]): Promise<CommunityInsight> {
  const content = posts.map((p) => `${p.author.username}: ${p.content}`).join("\n\n")

  const { object } = await generateObject({
    model: "openai/gpt-4o-mini",
    schema: z.object({
      overall_sentiment: z.enum(["positive", "neutral", "negative", "mixed"]),
      confidence: z.number().min(0).max(1),
      key_emotions: z.array(z.string()).max(5),
      tone_shift: z.string().optional(),
      reasoning: z.string(),
    }),
    prompt: `Analyze the sentiment and emotional tone of this forum thread.

Thread posts:
${content.slice(0, 4000)}

Identify:
1. Overall sentiment (positive/neutral/negative/mixed)
2. Key emotions expressed (e.g., frustration, excitement, confusion)
3. Any significant tone shifts during the conversation
4. Confidence in your assessment

Be specific and cite examples from the posts.`,
  })

  return {
    type: "sentiment",
    title: `${object.overall_sentiment.charAt(0).toUpperCase() + object.overall_sentiment.slice(1)} Discussion`,
    description: object.reasoning,
    confidence: object.confidence,
    metadata: {
      sentiment: object.overall_sentiment,
      emotions: object.key_emotions,
      tone_shift: object.tone_shift,
    },
  }
}

/**
 * Extract topics and themes
 * Showcases: Semantic understanding of forum content
 */
export async function extractTopics(thread: ForumsThread): Promise<CommunityInsight> {
  const content = `Title: ${thread.title}\n\n${thread.posts.map((p) => p.content).join("\n\n")}`

  const { object } = await generateObject({
    model: "openai/gpt-4o-mini",
    schema: z.object({
      main_topic: z.string(),
      subtopics: z.array(z.string()).max(5),
      related_technologies: z.array(z.string()).max(5),
      suggested_tags: z.array(z.string()).max(8),
      confidence: z.number().min(0).max(1),
    }),
    prompt: `Extract topics, themes, and relevant tags from this forum thread.

Thread content:
${content.slice(0, 5000)}

Identify:
1. Main topic or problem being discussed
2. Key subtopics or related issues
3. Technologies, tools, or frameworks mentioned
4. Suggested tags for categorization

Be concise and specific.`,
  })

  return {
    type: "topic",
    title: object.main_topic,
    description: `Subtopics: ${object.subtopics.join(", ")}`,
    confidence: object.confidence,
    metadata: {
      main_topic: object.main_topic,
      subtopics: object.subtopics,
      technologies: object.related_technologies,
      tags: object.suggested_tags,
    },
  }
}

/**
 * Identify user expertise levels
 * Showcases: Community understanding through AI
 */
export async function identifyExpertise(posts: ForumsPost[]): Promise<CommunityInsight> {
  const userPosts = posts.reduce(
    (acc, p) => {
      const user = p.author.username || p.author.name || "Anonymous"
      if (!acc[user]) acc[user] = []
      acc[user].push(p.content)
      return acc
    },
    {} as Record<string, string[]>,
  )

  const topContributors = Object.entries(userPosts)
    .map(([user, posts]) => ({
      user,
      post_count: posts.length,
      content: posts.join(" ").slice(0, 1000),
    }))
    .sort((a, b) => b.post_count - a.post_count)
    .slice(0, 3)

  const { object } = await generateObject({
    model: "openai/gpt-4o-mini",
    schema: z.object({
      experts: z.array(
        z.object({
          username: z.string(),
          expertise_level: z.enum(["beginner", "intermediate", "expert", "authority"]),
          specialty: z.string(),
          helpful_score: z.number().min(0).max(1),
        }),
      ),
      confidence: z.number().min(0).max(1),
      summary: z.string(),
    }),
    prompt: `Analyze user expertise based on their contributions to this thread.

Top contributors:
${topContributors.map((c) => `${c.user} (${c.post_count} posts): ${c.content}`).join("\n\n")}

For each user, determine:
1. Expertise level (beginner/intermediate/expert/authority)
2. Their specialty or area of knowledge
3. How helpful their contributions are (0-1 score)

Be fair and evidence-based.`,
  })

  return {
    type: "expertise",
    title: "Community Expertise",
    description: object.summary,
    confidence: object.confidence,
    metadata: {
      experts: object.experts,
    },
  }
}

/**
 * Generate smart reply suggestions
 * Showcases: AI-powered community interaction enhancement
 */
export async function suggestReplies(thread: ForumsThread): Promise<CommunityInsight> {
  const recentPosts = thread.posts.slice(-5)
  const content = recentPosts.map((p) => `${p.author.username}: ${p.content}`).join("\n\n")

  const { object } = await generateObject({
    model: "openai/gpt-4o-mini",
    schema: z.object({
      suggestions: z.array(
        z.object({
          type: z.enum(["question", "clarification", "solution", "resource", "encouragement"]),
          text: z.string(),
          rationale: z.string(),
        }),
      ),
      confidence: z.number().min(0).max(1),
    }),
    prompt: `Generate 3-5 helpful reply suggestions for this forum thread.

Thread title: ${thread.title}

Recent posts:
${content}

For each suggestion:
1. Choose a type (question, clarification, solution, resource, encouragement)
2. Write a natural, helpful reply
3. Explain why this reply would be valuable

Be constructive and community-focused.`,
  })

  return {
    type: "suggestion",
    title: "Smart Reply Suggestions",
    description: `${object.suggestions.length} AI-generated replies to help the discussion`,
    confidence: object.confidence,
    metadata: {
      suggestions: object.suggestions,
    },
  }
}

/**
 * Detect trends and patterns
 * Showcases: Multi-thread analysis capabilities
 */
export async function detectTrends(threads: ForumsThread[]): Promise<CommunityInsight> {
  const threadSummaries = threads.slice(0, 10).map((t) => ({
    title: t.title,
    post_count: t.posts.length,
    tags: t.tags || [],
    preview: t.posts[0]?.content.slice(0, 200),
  }))

  const { object } = await generateObject({
    model: "openai/gpt-4o-mini",
    schema: z.object({
      trending_topics: z.array(
        z.object({
          topic: z.string(),
          frequency: z.number(),
          growth: z.enum(["rising", "stable", "declining"]),
        }),
      ),
      emerging_issues: z.array(z.string()),
      common_questions: z.array(z.string()),
      confidence: z.number().min(0).max(1),
      summary: z.string(),
    }),
    prompt: `Analyze trends across these forum threads.

Threads:
${JSON.stringify(threadSummaries, null, 2)}

Identify:
1. Trending topics (with estimated frequency and growth trajectory)
2. Emerging issues or new problems
3. Common questions being asked
4. Overall community trends

Provide actionable insights for community managers.`,
  })

  return {
    type: "trend",
    title: "Community Trends",
    description: object.summary,
    confidence: object.confidence,
    metadata: {
      trending_topics: object.trending_topics,
      emerging_issues: object.emerging_issues,
      common_questions: object.common_questions,
    },
  }
}

/**
 * Comprehensive community insights
 * Combines all AI features for full picture
 */
export async function getComprehensiveInsights(thread: ForumsThread): Promise<CommunityInsight[]> {
  const [sentiment, topics, expertise, replies] = await Promise.all([
    analyzeSentiment(thread.posts),
    extractTopics(thread),
    identifyExpertise(thread.posts),
    suggestReplies(thread),
  ])

  return [sentiment, topics, expertise, replies]
}
