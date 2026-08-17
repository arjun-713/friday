"use client";

import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import type { ComponentType, SVGProps } from "react";
import {
  ArrowPathIcon,
  ArrowRightIcon,
  ArrowTopRightOnSquareIcon,
  BookOpenIcon,
  CheckIcon,
  ChevronDownIcon,
  ClipboardDocumentIcon,
  ComputerDesktopIcon,
  EllipsisHorizontalIcon,
  HandThumbDownIcon,
  HandThumbUpIcon,
  MicrophoneIcon,
  PaperAirplaneIcon,
  PaperClipIcon,
  PauseIcon,
  PrinterIcon,
  UserCircleIcon,
  WifiIcon,
} from "@heroicons/react/24/outline";
import {
  API_BASE_URL,
  deleteDiagnosticSession,
  troubleshootStream,
  TroubleshootingApiError,
  type DiagnosticOption,
  type TroubleshootingResponse,
} from "../lib/api";
import { FridayVoiceClient, type VoiceEvent } from "../lib/voice";

type Message = { id: string; role: "user" | "assistant"; text: string; meta?: string; response?: TroubleshootingResponse };
type SessionState = "ready" | "listening" | "thinking" | "speaking" | "interrupted";
type DeviceCategory = "laptop" | "router" | "printer";
type SessionStatus = "active" | "open" | "resolved";

type Session = {
  id: string;
  title: string;
  device: string;
  category: DeviceCategory;
  status: SessionStatus;
  createdAt: string;
  updatedAt: string;
  messages: Message[];
  selectedAnswer: string | null;
};

const SESSION_STORAGE_KEY = "friday.troubleshooting-sessions.v1";

function relativeSessionTime(value: string): string {
  const elapsed = Math.max(0, Date.now() - new Date(value).getTime());
  const minutes = Math.floor(elapsed / 60_000);
  if (minutes < 1) return "Now";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

function restoreSessions(raw: unknown): Session[] {
  if (!Array.isArray(raw)) return [];
  const fallbackTimestamp = new Date().toISOString();
  return raw.flatMap((item): Session[] => {
    if (!item || typeof item !== "object") return [];
    const session = item as Partial<Session>;
    if (typeof session.id !== "string" || typeof session.title !== "string" || typeof session.device !== "string") return [];
    if (session.category !== "laptop" && session.category !== "router" && session.category !== "printer") return [];
    return [{
      id: session.id,
      title: session.title,
      device: session.device,
      category: session.category,
      status: session.status === "resolved" ? "resolved" : session.status === "open" ? "open" : "active",
      createdAt: typeof session.createdAt === "string" ? session.createdAt : fallbackTimestamp,
      updatedAt: typeof session.updatedAt === "string" ? session.updatedAt : fallbackTimestamp,
      messages: Array.isArray(session.messages) ? session.messages : [],
      selectedAnswer: typeof session.selectedAnswer === "string" ? session.selectedAnswer : null,
    }];
  });
}

function streamedInstruction(json: string): string {
  const match = json.match(/"instruction"\s*:\s*"((?:\\.|[^"\\])*)/);
  if (!match) return "";
  try {
    return JSON.parse(`"${match[1]}"`) as string;
  } catch {
    return "";
  }
}

const deviceProfiles = {
  laptop: { name: "ThinkPad T480", detail: "Laptop / Desktop", icon: "laptop" as const },
  router: { name: "TP-Link Archer C6", detail: "Wi-Fi Router", icon: "router" as const },
  printer: { name: "LaserJet Pro M404/M405", detail: "Printer", icon: "printer" as const },
};

type IconName = "arrow" | "mic" | "send" | "check" | "pause" | "chevron" | "laptop" | "router" | "printer" | "external" | "manual" | "more" | "paperclip" | "user" | "regenerate" | "like" | "dislike" | "copy";
const iconMap: Record<IconName, ComponentType<SVGProps<SVGSVGElement>>> = {
  arrow: ArrowRightIcon,
  mic: MicrophoneIcon,
  send: PaperAirplaneIcon,
  check: CheckIcon,
  pause: PauseIcon,
  chevron: ChevronDownIcon,
  laptop: ComputerDesktopIcon,
  router: WifiIcon,
  printer: PrinterIcon,
  external: ArrowTopRightOnSquareIcon,
  manual: BookOpenIcon,
  more: EllipsisHorizontalIcon,
  paperclip: PaperClipIcon,
  user: UserCircleIcon,
  regenerate: ArrowPathIcon,
  like: HandThumbUpIcon,
  dislike: HandThumbDownIcon,
  copy: ClipboardDocumentIcon,
};

function Icon({ name }: { name: IconName }) {
  const Component = iconMap[name];
  return <Component aria-hidden="true" className="icon" />;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [state, setState] = useState<SessionState>("ready");
  const [activeSession, setActiveSession] = useState("new");
  const [sessionId, setSessionId] = useState("session-initial");
  const [caseQuery, setCaseQuery] = useState("");
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionsHydrated, setSessionsHydrated] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<DeviceCategory>("router");
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [whyOpen, setWhyOpen] = useState(false);
  const [sessionMenuOpen, setSessionMenuOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [attachment, setAttachment] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<"like" | "dislike" | null>(null);
  const [copied, setCopied] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const requestController = useRef<AbortController | null>(null);
  const messageSequence = useRef(0);
  const voiceClient = useRef<FridayVoiceClient | null>(null);
  const voiceAssistantId = useRef<string | null>(null);
  const voiceStreamedJson = useRef("");

  function createMessageId(role: "user" | "assistant"): string {
    messageSequence.current += 1;
    // A timestamp alone can collide when a final voice transcript creates the
    // user and assistant messages in the same render. Use a UUID so IDs remain
    // stable even across Next.js Fast Refresh boundaries.
    const uniquePart = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${messageSequence.current}`;
    return `${role}-${uniquePart}`;
  }

  const selectedDevice = deviceProfiles[selectedCategory];

  function startNewSession(category = selectedCategory) {
    requestController.current?.abort();
    void stopVoice();
    setSelectedCategory(category);
    setActiveSession("new");
    const nextSessionId = `session-${globalThis.crypto?.randomUUID?.() ?? Date.now()}`;
    setSessionId(nextSessionId);
    setCaseQuery("");
    setMessages([]);
    setSelectedAnswer(null);
    setState("ready");
    setApiError(null);
    setWhyOpen(false);
    setEvidenceOpen(false);
    setSessionMenuOpen(false);
  }

  function chooseSession(session: Session) {
    requestController.current?.abort();
    void stopVoice();
    setActiveSession(session.id);
    setSessionId(session.id);
    setCaseQuery(session.title);
    setSelectedCategory(session.category);
    setMessages(session.messages);
    setSelectedAnswer(session.selectedAnswer);
    setState("ready");
    setApiError(null);
    setWhyOpen(false);
    setEvidenceOpen(false);
  }

  async function deleteCurrentSession() {
    const deletingSession = activeSession;
    if (activeSession !== "new") setSessions((current) => current.filter((session) => session.id !== activeSession));
    startNewSession();
    if (deletingSession === "new") return;
    try {
      await deleteDiagnosticSession(deletingSession);
    } catch {
      // The chat is already removed from this browser. A later server cleanup can remove stale state.
    }
  }

  async function runTroubleshoot(
    query: string,
    displayText = query,
    addUser = true,
    interaction: { observation?: string; selectedOption?: string } = {},
  ) {
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setApiError(null);
    setState("thinking");
    if (addUser) {
      setMessages((current) => [...current, { id: createMessageId("user"), role: "user", text: displayText, meta: "just now" }]);
      if (!caseQuery) {
        setCaseQuery(displayText);
        setActiveSession(sessionId);
        const timestamp = new Date().toISOString();
        setSessions((current) => [{ id: sessionId, title: displayText, device: selectedDevice.name, category: selectedCategory, status: "active", createdAt: timestamp, updatedAt: timestamp, messages: [], selectedAnswer: null }, ...current]);
      }
    }

    const manufacturer = selectedCategory === "router" ? "TP-Link" : selectedCategory === "printer" ? "HP" : "Lenovo";
    try {
      const assistantId = createMessageId("assistant");
      setMessages((current) => [...current, { id: assistantId, role: "assistant", text: "", meta: "just now" }]);
      let streamedJson = "";
      let completed: TroubleshootingResponse | null = null;
      await troubleshootStream(
        {
          query,
          manufacturer,
          model: selectedDevice.name,
          session_id: sessionId,
          observation: interaction.observation,
          selected_option: interaction.selectedOption,
        },
        (event) => {
          if (event.type === "token") {
            streamedJson += event.text;
            const preview = streamedInstruction(streamedJson);
            if (preview) {
              setMessages((current) => current.map((message) => (message.id === assistantId ? { ...message, text: preview } : message)));
            }
          }
          if (event.type === "complete") {
            completed = event.response;
            const result = event.response;
            const answerText =
              result.status === "ready"
                ? result.step?.instruction ?? result.answer ?? "The manual does not provide an answer for this observation."
                : result.missing_observations.length > 0
                  ? `I need one more observation: ${result.missing_observations.join(", ")}.`
                  : "I could not verify a safe next step from the available manuals.";
            setMessages((current) =>
              current.map((message) => (message.id === assistantId ? { ...message, text: answerText, response: result } : message)),
            );
          }
        },
        controller.signal,
      );
      if (!completed) throw new TroubleshootingApiError("The troubleshooting stream ended without a response.", 502);
      setState("ready");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setApiError(
        error instanceof TroubleshootingApiError
          ? error.message
          : "The request failed before Friday received a response. Check that the backend is running on port 8000.",
      );
      setMessages((current) => current.filter((message) => message.text || message.role === "user"));
      setState("interrupted");
    }
  }

  function submitAnswer(option: DiagnosticOption) {
    setSelectedAnswer(option.label);
    void runTroubleshoot(
      `${caseQuery || selectedDevice.name} Observation: ${option.label}`,
      option.label,
      true,
      { observation: option.label, selectedOption: option.id },
    );
  }

  function submitMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = draft.trim();
    if (!text) return;
    void runTroubleshoot(
      attachment ? `${text} Attached file: ${attachment}.` : text,
      attachment ? `${text} · ${attachment}` : text,
      true,
      { observation: text },
    );
    setDraft("");
    setAttachment(null);
  }

  function handleAttachment(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) setAttachment(file.name);
  }

  function regenerateResponse() {
    const latestUserMessage = [...messages].reverse().find((message) => message.role === "user");
    if (latestUserMessage) void runTroubleshoot(latestUserMessage.text, "", false);
  }

  async function copyResponse(response?: TroubleshootingResponse) {
    if (!response) return;
    const citations = response.citations.map((citation) => `${citation.document_title}, page ${citation.page}, ${citation.section}`).join("; ");
    await navigator.clipboard?.writeText(`${response.answer ?? ""}${citations ? `\n\nSources: ${citations}` : ""}`);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  function setAssistantPreview(id: string, text: string) {
    const preview = streamedInstruction(text);
    if (preview) {
      setMessages((current) => current.map((message) => (message.id === id ? { ...message, text: preview } : message)));
    }
  }

  function completeAssistant(id: string, response: TroubleshootingResponse) {
    const answerText =
      response.status === "ready"
        ? response.step?.instruction ?? response.answer ?? "The manual does not provide an answer for this observation."
        : response.missing_observations.length > 0
          ? `I need one more observation: ${response.missing_observations.join(", ")}.`
          : "I could not verify a safe next step from the available manuals.";
    setMessages((current) => current.map((message) => (message.id === id ? { ...message, text: answerText, response } : message)));
  }

  function handleVoiceEvent(event: VoiceEvent) {
    if (event.type === "speech.start") {
      requestController.current?.abort();
      setState("listening");
      return;
    }
    if (event.type === "transcript.partial") {
      setDraft(event.text);
      setState("listening");
      return;
    }
    if (event.type === "transcript.final") {
      const assistantId = createMessageId("assistant");
      voiceAssistantId.current = assistantId;
      voiceStreamedJson.current = "";
      setDraft("");
      setSelectedAnswer(null);
      setApiError(null);
      setMessages((current) => [
        ...current,
        { id: createMessageId("user"), role: "user", text: event.text, meta: "just now" },
        { id: assistantId, role: "assistant", text: "", meta: "just now" },
      ]);
      if (!caseQuery) {
        setCaseQuery(event.text);
        setActiveSession(sessionId);
        const timestamp = new Date().toISOString();
        setSessions((current) => [{ id: sessionId, title: event.text, device: selectedDevice.name, category: selectedCategory, status: "active", createdAt: timestamp, updatedAt: timestamp, messages: [], selectedAnswer: null }, ...current]);
      }
      setState("thinking");
      return;
    }
    if (event.type === "assistant.token" && voiceAssistantId.current) {
      voiceStreamedJson.current += event.text;
      setAssistantPreview(voiceAssistantId.current, voiceStreamedJson.current);
      return;
    }
    if (event.type === "assistant.complete" && voiceAssistantId.current) {
      completeAssistant(voiceAssistantId.current, event.response);
      setState(event.response.status === "ready" ? "speaking" : "listening");
      return;
    }
    if (event.type === "assistant.audio_complete") {
      setState("listening");
      return;
    }
    if (event.type === "assistant.cancelled") {
      const id = voiceAssistantId.current;
      if (id) setMessages((current) => current.filter((message) => message.id !== id || Boolean(message.response)));
      voiceAssistantId.current = null;
      voiceStreamedJson.current = "";
      setState("listening");
      return;
    }
    if (event.type === "voice.error") {
      setApiError(event.message);
      setState("interrupted");
      return;
    }
    if (event.type === "voice.closed") {
      voiceClient.current = null;
      setApiError(event.message);
      setState("interrupted");
    }
  }

  async function startVoice() {
    try {
      if (voiceClient.current) return;
      const client = new FridayVoiceClient(handleVoiceEvent);
      voiceClient.current = client;
      setState("listening");
      await client.start({ sessionId, manufacturer: selectedCategory === "router" ? "TP-Link" : selectedCategory === "printer" ? "HP" : "Lenovo", model: selectedDevice.name });
      setApiError(null);
      setState("listening");
    } catch (error) {
      voiceClient.current = null;
      setApiError(error instanceof Error ? error.message : "Could not start the microphone.");
      setState("interrupted");
    }
  }

  async function stopVoice() {
    const client = voiceClient.current;
    voiceClient.current = null;
    await client?.stop();
    setDraft("");
    setState("ready");
  }

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(SESSION_STORAGE_KEY);
      if (stored) setSessions(restoreSessions(JSON.parse(stored)));
    } catch {
      window.localStorage.removeItem(SESSION_STORAGE_KEY);
    } finally {
      setSessionsHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!sessionsHydrated) return;
    window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessions));
  }, [sessions, sessionsHydrated]);

  useEffect(() => {
    if (!sessionsHydrated || activeSession === "new") return;
    setSessions((current) => current.map((session) => session.id === activeSession ? {
      ...session,
      title: caseQuery || session.title,
      device: selectedDevice.name,
      category: selectedCategory,
      updatedAt: new Date().toISOString(),
      messages,
      selectedAnswer,
    } : session));
  }, [activeSession, caseQuery, messages, selectedAnswer, selectedCategory, selectedDevice.name, sessionsHydrated]);

  useEffect(() => {
    return () => {
      requestController.current?.abort();
      void voiceClient.current?.stop();
    };
  }, []);

  const isListening = state === "listening";
  const isThinking = state === "thinking";
  const voiceConnected = voiceClient.current !== null;
  const latestAssistantMessage = [...messages].reverse().find((message) => message.role === "assistant" && message.response);
  const latestResponse = latestAssistantMessage?.response;
  const latestCitation = latestResponse?.citations[0];
  const activeQuestion = latestResponse?.status === "ready" ? latestResponse.step?.question : undefined;
  const observations = selectedAnswer ? [selectedAnswer] : [];
  const orderedSessions = [...sessions].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="wordmark" href="#conversation" aria-label="Friday home"><span className="wordmark-mark"><Icon name="router" /></span><span>friday</span></a>
        <div className="topbar-center"><span className="topbar-context">{selectedDevice.name}<span>/</span>{caseQuery || "New session"}</span></div>
      </header>

      <div className="workspace">
        <aside className="session-sidebar" aria-label="Device and troubleshooting sessions">
          <div className="device-picker">
            <span className="sidebar-title">DEVICE CATEGORIES</span>
            {Object.entries(deviceProfiles).map(([category, device]) => (
              <button className={`device-category ${selectedCategory === category ? "selected" : ""}`} key={category} type="button" onClick={() => startNewSession(category as DeviceCategory)}>
                <Icon name={device.icon} /><span>{device.detail}</span>
              </button>
            ))}
          </div>

          <div className="current-device">
            <span className="sidebar-title">CURRENT DEVICE</span>
            <div className="current-device-card"><span className="current-device-image"><Icon name={selectedDevice.icon} /></span><div><strong>{selectedDevice.name}</strong><span>{selectedDevice.detail}</span><button type="button" onClick={() => startNewSession()}>Change</button></div></div>
          </div>

          <button className="new-session-quiet" type="button" onClick={() => startNewSession()}><span>＋</span> New session</button>
          <div className="sidebar-heading"><h2>Sessions</h2></div>
          <div className="session-list">
            {orderedSessions.map((session) => (
              <button className={`session-item ${activeSession === session.id ? "selected" : ""}`} key={session.id} type="button" onClick={() => chooseSession(session)}>
                <span className={`session-status-dot ${activeSession === session.id ? "active" : session.status}`} aria-hidden="true" />
                <span className="session-item-copy"><strong>{session.title}</strong><span>{session.device}</span></span>
                <span className="session-time">{session.status === "resolved" ? "Resolved" : relativeSessionTime(session.updatedAt)}</span>
              </button>
            ))}
            {sessionsHydrated && orderedSessions.length === 0 && <p className="session-empty">Your troubleshooting history will appear here.</p>}
          </div>
          <button className="profile-row" type="button" aria-label="Open profile"><span className="profile-avatar"><Icon name="user" /></span><span className="profile-copy"><strong>Profile</strong><span>Account settings</span></span><Icon name="chevron" /></button>
        </aside>

        <section className="conversation" id="conversation" aria-labelledby="conversation-title">
          <div className="troubleshooting-thread">
            <div className="conversation-header">
              <div><span className="case-context">{selectedDevice.detail.toUpperCase()} / {selectedDevice.name.toUpperCase()}</span><h1 id="conversation-title">{caseQuery || "New troubleshooting session"}</h1></div>
              <div className="session-menu-wrap">
                <button className="session-menu-button" type="button" aria-label="Session actions" aria-expanded={sessionMenuOpen} onClick={() => setSessionMenuOpen((open) => !open)}><Icon name="more" /></button>
                {sessionMenuOpen && <div className="session-menu" role="menu"><button type="button" onClick={() => startNewSession()}>Start a new session</button>{activeSession !== "new" && <button type="button" onClick={() => void deleteCurrentSession()}>Delete this session</button>}</div>}
              </div>
              <button className="evidence-toggle" type="button" aria-expanded={evidenceOpen} onClick={() => setEvidenceOpen((open) => !open)}>What we know <Icon name="chevron" /></button>
            </div>

            <div className="message-list" aria-live="polite">
              {messages.length === 0 && <div className="empty-thread"><h2>Describe the problem to start.</h2><p>Use your own words. Friday will ask for one observation at a time.</p></div>}
              {messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  {message.role === "user" && <div className="message-meta"><span>You</span><span>{message.meta}</span></div>}
                  {message.role === "user" && <p>{message.text}</p>}
                  {message.role === "assistant" && (
                    message.id !== latestAssistantMessage?.id ? (
                      <div className="assistant-history">
                        <p>{message.text || "Checking the manual…"}</p>
                        {message.response?.citations[0] && <a href={message.response.citations[0].source_url || "#source"}>{message.response.citations[0].document_title} · p. {message.response.citations[0].page}</a>}
                      </div>
                    ) : <div className={`step-panel ${message.response?.status === "abstained" ? "abstained-panel" : ""}`}>
                      <div className="step-heading"><h2>{message.response?.status === "abstained" ? (message.response.missing_observations.length > 0 ? "I need another observation" : "No verified step found") : message.response?.step?.title ?? "Next check"}</h2></div>
                      <p className="instruction response-copy">{message.text}</p>
                      {message.response?.status === "abstained" ? <ul className="missing-observations">{message.response.missing_observations.map((observation) => <li key={observation}>{observation}</li>)}</ul> : <>
                        {message.response?.step && <div className="procedure-content"><div className="procedure-copy"><p className="instruction">{message.response.step.question}</p></div></div>}
                        {message.response?.step && message.response.step.options.length > 0 && <div className="answer-options" aria-label="Diagnostic answer options">{message.response.step.options.map((option) => <button className={selectedAnswer === option.label ? "selected" : ""} key={option.id} type="button" aria-pressed={selectedAnswer === option.label} onClick={() => submitAnswer(option)}>{option.label}</button>)}</div>}
                        {message.response?.images && message.response.images.length > 0 && <div className="manual-images" aria-label="Figures from the manufacturer manual">{message.response.images.map((image) => <figure key={image.asset_id}><img src={`${API_BASE_URL}${image.url}`} alt={`${image.document_title}, page ${image.page}`} /><figcaption>{image.document_title} · p. {image.page}</figcaption></figure>)}</div>}
                        <button className="why-button" type="button" aria-expanded={whyOpen} onClick={() => setWhyOpen((open) => !open)}>Why this step?</button>
                        {whyOpen && <p className="why-copy">The retrieved manual evidence is used to choose the next observation before changing any settings.</p>}
                      </>}
                      {message.response?.citations[0] && <div className="source-line"><Icon name="manual" /><a href={message.response.citations[0].source_url || "#source"}>{message.response.citations[0].document_title} · p. {message.response.citations[0].page} · {message.response.citations[0].section}</a><Icon name="external" /></div>}
                    </div>
                  )}
                  {message.role === "assistant" && <div className="assistant-actions" aria-label="Response actions"><button type="button" aria-label="Regenerate response" title="Regenerate response" onClick={regenerateResponse}><Icon name="regenerate" /></button><button className={feedback === "like" ? "selected" : ""} type="button" aria-label="Like response" title="Like response" aria-pressed={feedback === "like"} onClick={() => setFeedback("like")}><Icon name="like" /></button><button className={feedback === "dislike" ? "selected" : ""} type="button" aria-label="Dislike response" title="Dislike response" aria-pressed={feedback === "dislike"} onClick={() => setFeedback("dislike")}><Icon name="dislike" /></button><button className={copied ? "selected" : ""} type="button" aria-label={copied ? "Response copied" : "Copy response"} title={copied ? "Copied" : "Copy response"} onClick={() => copyResponse(message.response)}><Icon name="copy" /></button></div>}
                </article>
              ))}
              {apiError && <div className="api-error" role="alert"><strong>Couldn&apos;t check the manuals.</strong><span>{apiError}</span><button type="button" onClick={() => { const latestUserMessage = [...messages].reverse().find((message) => message.role === "user"); if (latestUserMessage) void runTroubleshoot(latestUserMessage.text, "", false); }}>Try again</button></div>}
              {isThinking && <div className="thinking-line" role="status"><span className="thinking-pulse" /> Checking the manual</div>}
            </div>

            <div className="composer-wrap">
              {attachment && <div className="composer-attachment"><Icon name="paperclip" /><span>{attachment}</span><button type="button" aria-label="Remove attachment" onClick={() => setAttachment(null)}>×</button></div>}
              <form className={`composer ${isListening ? "listening" : ""}`} onSubmit={submitMessage}>
                <label className="sr-only" htmlFor="message">Describe what you see</label>
                <input id="message" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder={isListening ? "Voice input is ready; type if needed…" : "Describe what you see…"} />
                <input className="sr-only" id="attachment" type="file" accept="image/*,.pdf,.txt" onChange={handleAttachment} />
                <label className="attach-button" htmlFor="attachment"><Icon name="paperclip" /><span>Attach</span></label>
                {voiceConnected ? <button className="mic-button active" type="button" aria-label="Stop listening" onClick={() => void stopVoice()}><Icon name="pause" /></button> : <button className={`mic-button ${draft ? "quiet" : "primary"}`} type="button" aria-label="Start voice input" onClick={() => void startVoice()}><Icon name="mic" /></button>}
                <button className="send-button visible" type="submit" aria-label="Send observation" disabled={!draft.trim()}><Icon name="send" /></button>
              </form>
              {voiceConnected && <div className="voice-status" role="status"><span className="waveform" aria-hidden="true"><i /><i /><i /><i /><i /></span><span>{isListening ? "Listening — speak naturally; Friday will send the final transcript." : "Friday is responding — interrupt at any time."}</span><button type="button" onClick={() => void stopVoice()}>Stop</button></div>}
            </div>
          </div>
        </section>

        <aside className={`diagnostic-rail ${evidenceOpen ? "mobile-open" : ""}`} aria-label="Evidence ledger">
          <div className="rail-header"><h2>What we know</h2><button className="rail-toggle" type="button" aria-label="Collapse evidence ledger"><Icon name="chevron" /></button></div>
          <div className="rail-section"><div className="rail-label">DEVICE</div><p className="rail-device">{selectedDevice.name}<span>{selectedDevice.detail}</span></p></div>
          <div className="rail-section"><div className="rail-label">OBSERVED</div>{observations.length > 0 ? <ul className="observation-list">{observations.map((observation) => <li key={observation}><span className="observation-dot done" /><span>{observation}</span></li>)}</ul> : <p className="rail-empty">No confirmed observations yet.</p>}</div>
          {activeQuestion && <div className="rail-section"><div className="rail-label">NEED TO VERIFY</div><ul className="observation-list"><li><span className="observation-dot pending" /><span>{activeQuestion}</span></li></ul></div>}
          {latestCitation && <div className="rail-section evidence-section"><div className="rail-label">MANUAL EVIDENCE</div><div className="evidence-card"><strong>{latestCitation.document_title}</strong><span>Page {latestCitation.page} · {latestCitation.section}</span><a href={latestCitation.source_url || "#source"}>Open cited page <Icon name="arrow" /></a></div></div>}
        </aside>
      </div>
    </main>
  );
}
