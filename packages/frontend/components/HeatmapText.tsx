'use client';

interface HeatmapTextProps {
  tokens: string[];
  heatmap: number[];
}

function heatColor(score: number): string {
  const clamped = Math.max(-1, Math.min(1, score));
  if (clamped >= 0) {
    const alpha = 0.15 + clamped * 0.55;
    return `rgba(34, 197, 94, ${alpha})`;
  }
  const alpha = 0.15 + Math.abs(clamped) * 0.55;
  return `rgba(239, 68, 68, ${alpha})`;
}

export function HeatmapText({ tokens, heatmap }: HeatmapTextProps) {
  if (!tokens.length) {
    return <p className="text-sm text-slate-400">No token attributions available.</p>;
  }

  return (
    <div className="flex flex-wrap gap-2 leading-8">
      {tokens.map((token, index) => {
        const score = heatmap[index] ?? 0;
        return (
          <span
            key={`${token}-${index}`}
            title={`Attribution: ${score.toFixed(3)}`}
            className="rounded-md px-2 py-1 text-base text-slate-100"
            style={{ backgroundColor: heatColor(score) }}
          >
            {token}
          </span>
        );
      })}
    </div>
  );
}
