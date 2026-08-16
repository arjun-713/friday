export type TroubleshootingRequest = {
  query: string;
  manufacturer?: string;
  model?: string;
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
  status: "ready" | "abstained";
  answer?: string | null;
  evidence: EvidenceContext[];
  citations: Citation[];
  missing_observations: string[];
  retrieval: RetrievalSummary;
};

export class TroubleshootingApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "TroubleshootingApiError";
    this.status = status;
  }
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

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
