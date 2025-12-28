import { type NextRequest, NextResponse } from "next/server"
import { forumsClient } from "@/lib/forums/client"
import { getComprehensiveInsights } from "@/lib/ai/community-insights"

export async function GET(request: NextRequest, { params }: { params: { threadId: string } }) {
  const { threadId } = params

  try {
    // Fetch thread from Foru.ms
    const thread = await forumsClient.getThread(threadId)

    if (!thread || !thread.posts || thread.posts.length === 0) {
      return NextResponse.json({ error: "Thread not found or has no posts" }, { status: 404 })
    }

    // Generate AI insights
    const insights = await getComprehensiveInsights(thread)

    return NextResponse.json({
      success: true,
      threadId,
      insights,
      generatedAt: new Date().toISOString(),
    })
  } catch (error) {
    console.error("[v0] Error generating insights:", error)
    return NextResponse.json(
      {
        error: "Failed to generate insights",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
