export interface ExplainResponse {
  category: number;
  label: string;
  display_label: string;
  confidence: number;
  scores: Record<string, number>;
  tokens: string[];
  heatmap: number[];
  method: string;
}

export interface ChatResponse extends ExplainResponse {
  reply: string;
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://localhost:4000/api';

async function postJson<T>(path: string, text: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    cache: 'no-store',
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.message ?? `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function explainText(text: string) {
  return postJson<ExplainResponse>('/explain', text);
}

export function chatText(text: string) {
  return postJson<ChatResponse>('/chat', text);
}

export function predictText(text: string) {
  return postJson<Omit<ExplainResponse, 'tokens' | 'heatmap' | 'method'>>('/predict', text);
}
