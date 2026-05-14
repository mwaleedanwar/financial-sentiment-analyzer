import { NextRequest, NextResponse } from "next/server";

// SENTIMENT_API_BASE_URL must be set on Vercel. ngrok-skip-browser-warning avoids free-ngrok blocking server fetch.
export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  const base = process.env.SENTIMENT_API_BASE_URL?.replace(/\/$/, "");
  if (!base) {
    return NextResponse.json(
      {
        detail:
          "Missing SENTIMENT_API_BASE_URL. In Vercel → Settings → Environment Variables, set it to your ngrok https URL.",
      },
      { status: 503 },
    );
  }

  let body: string;
  try {
    body = await request.text();
  } catch {
    return NextResponse.json({ detail: "Invalid request body" }, { status: 400 });
  }

  const upstream = await fetch(`${base}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
    body,
  });

  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}
