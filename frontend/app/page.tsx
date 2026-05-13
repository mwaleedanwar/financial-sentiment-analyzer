"use client";

/**
 * Phase 3: Single-page demo for the FastAPI `/predict` endpoint.
 *
 * Configuration:
 *   Set `NEXT_PUBLIC_API_URL` in `.env.local` to your deployed API root, e.g.
 *   https://your-service.onrender.com
 *
 * During local development the default points at localhost:8000 (see `getApiBase`).
 */

import { FormEvent, useMemo, useState } from "react";

type SentimentLabel = "Negative" | "Positive" | "Neutral";

type PredictResponse = {
  label: SentimentLabel;
  confidence: number;
};

function getApiBase(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
  return raw.replace(/\/$/, "");
}

function sentimentStyles(label: SentimentLabel): string {
  switch (label) {
    case "Positive":
      return "border-emerald-200 bg-emerald-50 text-emerald-900";
    case "Negative":
      return "border-rose-200 bg-rose-50 text-rose-900";
    default:
      return "border-slate-200 bg-white text-slate-800";
  }
}

export default function HomePage() {
  const apiBase = useMemo(() => getApiBase(), []);
  const [text, setText] = useState(
    "Fed signals patience on rate cuts as inflation remains sticky.",
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const payload = await res.json().catch(() => null);
      if (!res.ok) {
        const detail =
          typeof payload?.detail === "string"
            ? payload.detail
            : JSON.stringify(payload?.detail ?? payload);
        throw new Error(detail || `Request failed (${res.status})`);
      }
      setResult(payload as PredictResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16">
      <header className="mb-10 border-b border-slate-200 pb-8">
        <p className="text-xs font-semibold uppercase tracking-wide text-brand-600">
          NLP coursework demo
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-900">
          Financial sentiment classifier
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-600">
          Fine-tuned DistilBERT model (3-class: Negative, Positive, Neutral). Enter a
          finance-related sentence and call the deployed FastAPI service.
        </p>
        <p className="mt-2 text-xs text-slate-500">
          API base: <span className="font-mono text-slate-700">{apiBase}</span>
        </p>
      </header>

      <form
        onSubmit={onSubmit}
        className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      >
        <label htmlFor="sentence" className="block text-sm font-medium text-slate-700">
          Financial text
        </label>
        <textarea
          id="sentence"
          name="sentence"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="mt-2 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 shadow-inner outline-none ring-brand-500 transition focus:border-brand-500 focus:bg-white focus:ring-2"
          placeholder="Paste a headline, tweet, or filing snippet…"
        />

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={loading || !text.trim()}
            className="inline-flex items-center justify-center rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {loading ? "Scoring…" : "Predict sentiment"}
          </button>
          {loading && (
            <span className="text-sm text-slate-500" aria-live="polite">
              Calling model API…
            </span>
          )}
        </div>
      </form>

      {error && (
        <section
          className="mt-6 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900"
          role="alert"
        >
          <p className="font-medium">Request error</p>
          <p className="mt-1 font-mono text-xs text-rose-800">{error}</p>
        </section>
      )}

      {result && !loading && (
        <section className="mt-8">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
            Model output
          </h2>
          <div
            className={`mt-3 flex flex-col gap-3 rounded-2xl border px-5 py-4 shadow-sm sm:flex-row sm:items-center sm:justify-between ${sentimentStyles(result.label)}`}
          >
            <div>
              <p className="text-xs font-medium uppercase text-slate-500">Predicted sentiment</p>
              <p className="text-2xl font-semibold">{result.label}</p>
            </div>
            <div className="sm:text-right">
              <p className="text-xs font-medium uppercase text-slate-500">Confidence</p>
              <p className="text-xl font-semibold tabular-nums">
                {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </section>
      )}

      <footer className="mt-auto pt-16 text-center text-xs text-slate-400">
        Built with Next.js App Router + TailwindCSS · pairs with FastAPI `/predict`
      </footer>
    </main>
  );
}
