'use client';

import { useMemo, useState } from 'react';
import { HeatmapText } from '@/components/HeatmapText';
import { chatText, explainText, type ChatResponse, type ExplainResponse } from '@/lib/api';

const SAMPLE_MESSAGES = [
  'I love you so much!',
  'I am terrified of losing everything.',
  'Get away from me, I hate this!',
];

type Mode = 'explain' | 'chat';

export default function HomePage() {
  const [text, setText] = useState(SAMPLE_MESSAGES[0]);
  const [mode, setMode] = useState<Mode>('explain');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExplainResponse | ChatResponse | null>(null);

  const sortedScores = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.scores).sort((a, b) => b[1] - a[1]);
  }, [result]);

  async function handleAnalyze(selectedMode: Mode) {
    setLoading(true);
    setError(null);
    setMode(selectedMode);

    try {
      const payload =
        selectedMode === 'chat' ? await chatText(text.trim()) : await explainText(text.trim());
      setResult(payload);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_#1e293b,_#020617_55%)] px-4 py-10">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <header className="panel">
          <p className="text-sm uppercase tracking-[0.25em] text-sky-300">GoEmotions RoBERTa XAI</p>
          <h1 className="mt-2 text-3xl font-bold text-white md:text-4xl">
            Fine-Tuned Sentiment Classifier with Token Heatmaps
          </h1>
          <p className="mt-3 max-w-3xl text-slate-300">
            Classify text into 7 emotion groups and inspect Integrated Gradients attribution
            scores token-by-token.
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="panel space-y-4">
            <label htmlFor="message" className="block text-sm font-medium text-slate-300">
              Message or question
            </label>
            <textarea
              id="message"
              className="input-area min-h-[160px] resize-y"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Type a sentence to classify..."
            />

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                className="primary-btn"
                disabled={loading || !text.trim()}
                onClick={() => handleAnalyze('explain')}
              >
                {loading && mode === 'explain' ? 'Analyzing...' : 'Explain with Heatmap'}
              </button>
              <button
                type="button"
                className="secondary-btn"
                disabled={loading || !text.trim()}
                onClick={() => handleAnalyze('chat')}
              >
                {loading && mode === 'chat' ? 'Thinking...' : 'Chatbot Classify'}
              </button>
            </div>

            <div className="flex flex-wrap gap-2">
              {SAMPLE_MESSAGES.map((sample) => (
                <button
                  key={sample}
                  type="button"
                  className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:border-sky-400"
                  onClick={() => setText(sample)}
                >
                  {sample}
                </button>
              ))}
            </div>

            {error ? <p className="text-sm text-red-400">{error}</p> : null}
          </div>

          <div className="panel space-y-4">
            <h2 className="text-lg font-semibold text-white">Prediction</h2>
            {result ? (
              <>
                <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
                  <p className="text-sm text-slate-400">Category {result.category}</p>
                  <p className="mt-1 text-xl font-semibold text-sky-300">{result.display_label}</p>
                  <p className="mt-2 text-sm text-slate-300">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </p>
                  {'reply' in result ? (
                    <p className="mt-3 text-sm leading-6 text-slate-200">{result.reply}</p>
                  ) : null}
                </div>

                <div className="space-y-3">
                  {sortedScores.map(([label, score]) => (
                    <div key={label}>
                      <div className="mb-1 flex justify-between text-xs text-slate-400">
                        <span>{label}</span>
                        <span>{(score * 100).toFixed(1)}%</span>
                      </div>
                      <div className="score-bar">
                        <div className="score-fill" style={{ width: `${score * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-slate-400">Run an analysis to see probabilities.</p>
            )}
          </div>
        </section>

        <section className="panel space-y-4">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-lg font-semibold text-white">Token Attribution Heatmap</h2>
            <span className="text-xs uppercase tracking-wide text-slate-400">
              Integrated Gradients
            </span>
          </div>
          {result ? (
            <HeatmapText tokens={result.tokens} heatmap={result.heatmap} />
          ) : (
            <p className="text-sm text-slate-400">
              Positive attribution highlights supportive tokens in green; negative attribution
              appears in red.
            </p>
          )}
        </section>
      </div>
    </main>
  );
}
