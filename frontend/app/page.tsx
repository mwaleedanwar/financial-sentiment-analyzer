"use client";

import { FormEvent, useMemo, useState } from "react";

type SentimentLabel = "Bull" | "Bear" | "Neutral";

type PredictResponse = {
  label: SentimentLabel;
  confidence: number;
};

type PredictConfig = {
  predictUrl: string;
  displayLabel: string;
};

function getPredictConfig(): PredictConfig {
  const useProxy = process.env.NEXT_PUBLIC_USE_VERCEL_PROXY === "1";
  const directBase = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

  if (useProxy) {
    return {
      predictUrl: "/api/predict",
      displayLabel: "Vercel proxy -> SENTIMENT_API_BASE_URL (ngrok)",
    };
  }

  return {
    predictUrl: `${directBase}/predict`,
    displayLabel: directBase,
  };
}

const SAMPLE_TEXTS = [
  "Fed signals patience on rate cuts as inflation remains sticky.",
  "Apple beats earnings expectations; shares rally in after-hours trading.",
  "Regional bank faces liquidity concerns amid deposit outflows.",
];

function sentimentTheme(label: SentimentLabel) {
  switch (label) {
    case "Bear":
      return {
        card: "border-emerald-500/30 bg-emerald-500/10",
        badge: "bg-emerald-400/20 text-emerald-300 ring-emerald-400/40",
        bar: "bg-emerald-400",
        glow: "shadow-[0_0_40px_-8px_rgba(52,211,153,0.45)]",
        icon: "↑",
      };
    case "Bull":
      return {
        card: "border-rose-500/30 bg-rose-500/10",
        badge: "bg-rose-400/20 text-rose-300 ring-rose-400/40",
        bar: "bg-rose-400",
        glow: "shadow-[0_0_40px_-8px_rgba(251,113,133,0.45)]",
        icon: "↓",
      };
    default:
      return {
        card: "border-slate-500/30 bg-slate-500/10",
        badge: "bg-slate-400/20 text-slate-300 ring-slate-400/40",
        bar: "bg-slate-400",
        glow: "shadow-[0_0_40px_-8px_rgba(148,163,184,0.35)]",
        icon: "→",
      };
  }
}

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

function PageBackground() {
  return (
    <>
      <PageBackgroundBlobs />
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.35]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)
          `,
          backgroundSize: "48px 48px",
          maskImage: "radial-gradient(ellipse 80% 60% at 50% 0%, black, transparent)",
        }}
      />
    </>
  );
}

function PageBackgroundBlobs() {
  return (
    <>
      <div className="pointer-events-none absolute -left-32 top-0 h-96 w-96 rounded-full bg-amber-500/10 blur-3xl" />
      <PageBackgroundBlobsRight />
      <div className="pointer-events-none absolute bottom-0 left-1/2 h-64 w-[32rem] -translate-x-1/2 rounded-full bg-emerald-600/5 blur-3xl" />
    </>
  );
}

function PageBackgroundBlobsRight() {
  return <div className="pointer-events-none absolute -right-24 top-1/3 h-80 w-80 rounded-full bg-indigo-600/15 blur-3xl" />;
}

function SampleChips({ text, onPick }: { text: string; onPick: (s: string) => void }) {
  return (
    <div className="mt-4 flex flex-wrap gap-2">
      {SAMPLE_TEXTS.map((sample) => (
        <button
          key={sample}
          type="button"
          onClick={() => onPick(sample)}
          className={`rounded-lg border px-2.5 py-1 text-left text-xs transition ${
            text === sample
              ? "border-amber-400/40 bg-amber-400/10 text-amber-200/90"
              : "border-white/5 bg-white/[0.03] text-slate-500 hover:border-amber-400/30 hover:bg-amber-400/5 hover:text-amber-200/90"
          }`}
        >
          {sample.slice(0, 42)}…
        </button>
      ))}
    </div>
  );
}

function ResultCard({ theme, label, confidencePct }: { theme: ReturnType<typeof sentimentTheme>; label: SentimentLabel; confidencePct: number }) {
  return (
    <div className={`mt-4 overflow-hidden rounded-2xl border p-6 backdrop-blur-sm ${theme.card} ${theme.glow}`}>
      <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-4">
          <span className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl text-xl font-bold ring-1 ${theme.badge}`} aria-hidden>
            {theme.icon}
          </span>
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Predicted sentiment</p>
            <p className="mt-1 text-3xl font-semibold tracking-tight text-white">{label}</p>
          </div>
        </div>
        <div className="w-full sm:w-48">
          <div className="flex items-baseline justify-between">
            <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Confidence</p>
            <p className="text-2xl font-semibold tabular-nums text-white">{confidencePct}%</p>
          </div>
          <ConfidenceBar theme={theme} confidencePct={confidencePct} />
        </div>
      </div>
    </div>
  );
}

function ConfidenceBar({ theme, confidencePct }: { theme: ReturnType<typeof sentimentTheme>; confidencePct: number }) {
  return (
    <div className="mt-2 h-2 overflow-hidden rounded-full bg-black/40">
      <div className={`h-full rounded-full transition-all duration-700 ease-out ${theme.bar}`} style={{ width: `${confidencePct}%` }} />
    </div>
  );
}

export default function HomePage() {
  const { predictUrl } = useMemo(() => getPredictConfig(), []);
  const [text, setText] = useState(SAMPLE_TEXTS[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await fetch(predictUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const payload = await res.json().catch(() => null);
      if (!res.ok) {
        const detail = typeof payload?.detail === "string" ? payload.detail : JSON.stringify(payload?.detail ?? payload);
        throw new Error(detail || `Request failed (${res.status})`);
      }
      setResult(payload as PredictResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const theme = result ? sentimentTheme(result.label) : null;
  const confidencePct = result ? Math.round(result.confidence * 1000) / 10 : 0;

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#070b14] text-slate-100">
      <PageBackground />

      <main className="relative z-10 mx-auto flex min-h-screen max-w-2xl flex-col px-5 py-12 sm:px-6 sm:py-16">
        <header className="mb-10">
          <HeaderBadge />
          <h1 className="mt-5 font-[family-name:var(--font-serif)] text-4xl font-medium tracking-tight text-white sm:text-5xl">
            Financial sentiment
            <span className="block bg-gradient-to-r from-amber-200 via-amber-400 to-amber-200 bg-clip-text text-transparent">classifier</span>
          </h1>
          <p className="mt-4 max-w-lg text-base leading-relaxed text-slate-400">
            Fine-tuned DistilBERT · Bull, Bear, Neutral. Paste a headline, tweet, or filing snippet and score it live.
          </p>
        </header>

        <form onSubmit={onSubmit} className="rounded-2xl border border-white/10 bg-white/[0.04] p-6 shadow-2xl shadow-black/40 backdrop-blur-md sm:p-7">
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="sentence" className="text-sm font-medium text-slate-300">
              Financial text
            </label>
            <span className="rounded-full bg-white/5 px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-slate-500">{text.length} chars</span>
          </div>

          <textarea
            id="sentence"
            name="sentence"
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="mt-3 w-full resize-none rounded-xl border border-white/10 bg-black/30 px-4 py-3.5 text-sm leading-relaxed text-slate-100 placeholder:text-slate-600 outline-none transition focus:border-amber-400/50 focus:ring-2 focus:ring-amber-400/20"
            placeholder="Paste a headline, tweet, or filing snippet…"
          />

          <SampleChips text={text} onPick={setText} />

          <div className="mt-6 flex flex-wrap items-center gap-4">
            <button
              type="submit"
              disabled={loading || !text.trim()}
              className="group relative inline-flex items-center justify-center overflow-hidden rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 px-6 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-amber-900/30 transition hover:from-amber-400 hover:to-amber-500 disabled:cursor-not-allowed disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-400 disabled:shadow-none"
            >
              <span className="relative z-10 flex items-center gap-2">
                {loading ? (
                  <>
                    <Spinner />
                    Scoring…
                  </>
                ) : (
                  <>
                    Predict sentiment
                    <span className="transition group-hover:translate-x-0.5" aria-hidden>
                      →
                    </span>
                  </>
                )}
              </span>
            </button>
            {loading && (
              <span className="text-sm text-slate-500" aria-live="polite">
                Calling model API…
              </span>
            )}
          </div>
        </form>

        {error && (
          <section className="mt-6 rounded-xl border border-rose-500/30 bg-rose-950/50 px-4 py-3.5 backdrop-blur-sm" role="alert">
            <p className="text-sm font-medium text-rose-300">Request error</p>
            <p className="mt-1.5 font-mono text-xs leading-relaxed text-rose-200/80">{error}</p>
          </section>
        )}

        {result && !loading && theme && (
          <section className="mt-8">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Model output</p>
            <ResultCard theme={theme} label={result.label} confidencePct={confidencePct} />
          </section>
        )}

        <footer className="mt-auto pt-14 text-center text-[11px] text-slate-600">Next.js | Tailwind | FastAPI</footer>
      </main>
    </div>
  );
}

function HeaderBadge() {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-amber-400/20 bg-amber-400/10 px-3 py-1 text-xs font-medium text-amber-200/90">
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-400" />
      </span>
      DistilBERT · 3-class NLP demo
    </div>
  );
}
