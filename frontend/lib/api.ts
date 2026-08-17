export type TroubleshootingRequest = {
  query: string;
  manufacturer?: string;
  model?: string;
  session_id?: string;
  observation?: string;
  selected_option?: string;
};

export type DiagnosticOption = {
  id: string;
  label: string;
};

export type DiagnosticStep = {
  step_id: string;
  title: string;
  instruction: string;
  question: string;
  options: DiagnosticOption[];
  source_ids: string[];
};

export type ManualImage = {
  asset_id: string;
  url: string;
  mime_type: string;
  document_title: string;
  page: number;
};

export type Citation = {
  chunk_id: string;
  document_id: string;
  document_title: string;
  manufacturer: string;
  model: string;
  document_version: string;
  page: number;
  section: string;
  source_url: string;
};

export type EvidenceContext = {
  chunk_id: string;
  content: string;
  section: string;
  pages: number[];
  citation: Citation;
};

export type RetrievalSummary = {
  abstained: boolean;
  reason?: string | null;
  timings_ms: Record<string, number>;
};

export type TroubleshootingResponse = {
  session_id: string;
  status: "ready" | "abstained";
  answer?: string | null;
  step?: DiagnosticStep | null;
  awaiting_observation: boolean;
  images: ManualImage[];
  evidence: EvidenceContext[];
  citations: Citation[];
  missing_observations: string[];
  retrieval: RetrievalSummary;
};

export type TroubleshootingStreamEvent =
  | { type: "retrieval"; retrieval: RetrievalSummary }
  | { type: "token"; text: string }
  | { type: "complete"; response: TroubleshootingResponse }
  | { type: "error"; message: string };

export class TroubleshootingApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "TroubleshootingApiError";
    this.status = status;
  }
}

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function troubleshoot(request: TroubleshootingRequest, signal?: AbortSignal): Promise<TroubleshootingResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/v1/troubleshoot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new TroubleshootingApiError("The troubleshooting service could not be reached.", 0);
  }

  if (!response.ok) {
    let detail = "The troubleshooting service returned an error.";
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      // Keep the stable user-facing error when the server response is not JSON.
    }
    throw new TroubleshootingApiError(detail, response.status);
  }

  return (await response.json()) as TroubleshootingResponse;
}

export async function troubleshootStream(
  request: TroubleshootingRequest,
  onEvent: (event: TroubleshootingStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/v1/troubleshoot/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(request),
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new TroubleshootingApiError("The troubleshooting service could not be reached.", 0);
  }

  if (!response.ok || !response.body) {
    throw new TroubleshootingApiError("The troubleshooting stream could not be opened.", response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      const data = event
        .split("\n")
        .find((line) => line.startsWith("data: "))
        ?.slice("data: ".length);
      if (!data) continue;
      const parsed = JSON.parse(data) as TroubleshootingStreamEvent;
      if (parsed.type === "error") throw new TroubleshootingApiError(parsed.message, 503);
      onEvent(parsed);
    }
    if (done) break;
  }
}
