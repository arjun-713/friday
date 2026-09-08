"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import type { ComponentType, SVGProps } from "react";
import {
  ArrowRightIcon,
  ArrowTopRightOnSquareIcon,
  BookOpenIcon,
  CheckIcon,
  ChevronDownIcon,
  ComputerDesktopIcon,
  EllipsisHorizontalIcon,
  MicrophoneIcon,
  PaperAirplaneIcon,
  PauseIcon,
  PrinterIcon,
  StopIcon,
  WifiIcon,
  ArrowPathIcon,
  DocumentDuplicateIcon,
  PlusIcon,
  XMarkIcon,
  Bars3Icon,
} from "@heroicons/react/24/outline";
import Brand, { BrandMark } from "./Brand";
import {
  API_BASE_URL,
  deleteDiagnosticSession,
  getSupportedDevices,
  troubleshootStream,
  TroubleshootingApiError,
  type DiagnosticOption,
  type SupportedDevice,
  type TroubleshootingResponse,
} from "../lib/api";
import { FridayVoiceClient, type VoiceEvent } from "../lib/voice";

type Message = { id: string; role: "user" | "assistant"; text: string; meta?: string; response?: TroubleshootingResponse };
// Full-duplex voice states: IDLE (ready), LISTENING, THINKING, SPEAKING,
// INTERRUPTING, RECOVERING, plus connecting/transient UI states.
type SessionState = "ready" | "connecting" | "listening" | "thinking" | "speaking" | "interrupting" | "recovering" | "interrupted" | "error";
type DeviceCategory = "laptop" | "router" | "printer";
type SessionStatus = "active" | "open" | "resolved";
type DiagnosticMode = "advance" | "clarify" | "solve" | "abstain";

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
  const restored = raw.flatMap((item): Session[] => {
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
  const byId = new Map<string, Session>();
  for (const session of restored) {
    const previous = byId.get(session.id);
    if (!previous || previous.updatedAt <= session.updatedAt) byId.set(session.id, session);
  }
  return [...byId.values()];
}

type DeviceProfile = { manufacturer: string; name: string; category: DeviceCategory; detail: string; icon: IconName };
const deviceCategories: Record<DeviceCategory, { label: string; icon: IconName }> = {
  laptop: { label: "Laptop / Desktop", icon: "laptop" },
  router: { label: "Wi-Fi Router", icon: "router" },
  printer: { label: "Printer", icon: "printer" },
};
const fallbackDeviceCatalog: DeviceProfile[] = [
  { manufacturer: "TP-Link", name: "Archer C6", category: "router", detail: deviceCategories.router.label, icon: deviceCategories.router.icon },
];

function deviceProfile(device: SupportedDevice): DeviceProfile {
  return {
    manufacturer: device.manufacturer,
    name: device.model,
    category: device.category,
    detail: deviceCategories[device.category].label,
    icon: deviceCategories[device.category].icon,
  };
}

type IconName = "arrow" | "mic" | "send" | "check" | "pause" | "stop" | "chevron" | "laptop" | "router" | "printer" | "external" | "manual" | "more";
const iconMap: Record<IconName, ComponentType<SVGProps<SVGSVGElement>>> = {
  arrow: ArrowRightIcon,
  mic: MicrophoneIcon,
  send: PaperAirplaneIcon,
  check: CheckIcon,
  pause: PauseIcon,
  stop: StopIcon,
  chevron: ChevronDownIcon,
  laptop: ComputerDesktopIcon,
  router: WifiIcon,
  printer: PrinterIcon,
  external: ArrowTopRightOnSquareIcon,
  manual: BookOpenIcon,
  more: EllipsisHorizontalIcon,
};

function Icon({ name }: { name: IconName }) {
  const Component = iconMap[name];
  return <Component aria-hidden="true" className="icon" />;
}

function responseText(response: TroubleshootingResponse): string {
  const raw = response.turn?.response
    ?? (response.status === "ready"
      ? response.step?.instruction ?? response.answer ?? "The manual does not provide an answer for this observation."
      : response.answer ?? "I could not verify a safe next step from the available manuals.");
  // Citations are rendered in the evidence row. Remove citation markers that
  // older/provider-specific response formats may have embedded in prose.
  // This also strips raw [source:...] markers so they never leak into chat or TTS.
  const withoutSourceMarkers = raw.replace(/\[source:[^\]]+\]/g, "");
  const withoutInlineCitations = response.citations.reduce((text, citation) => {
    const labels = [
      `[${citation.document_title} · p. ${citation.page} · ${citation.section}]`,
      `[${citation.document_title}, page ${citation.page}]`,
    ];
    return labels.reduce((value, label) => value.replaceAll(label, ""), text);
  }, withoutSourceMarkers);
  return withoutInlineCitations.replace(/[ \t]{2,}/g, " ").trim();
}

function sessionStatusLabel(status: SessionStatus): string {
  return status === "resolved" ? "Resolved" : "In progress";
}

function factKeyLabel(key: string): string {
  return key.replaceAll("_", " ");
}

function modePresentation(response: TroubleshootingResponse): { mode: DiagnosticMode; label: string; className: string } {
  if (response.status === "abstained") return { mode: "abstain", label: "MANUAL EVIDENCE INSUFFICIENT", className: "mode-abstain" };
  const mode = response.turn?.mode ?? "advance";
  if (mode === "solve") return { mode, label: "RESOLUTION", className: "mode-solve" };
  if (mode === "clarify") return { mode, label: "NEED ONE DETAIL", className: "mode-clarify" };
  return { mode: "advance", label: "NEXT CHECK", className: "mode-advance" };
}

export default function Casebook() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [state, setState] = useState<SessionState>("ready");
  const [activeSession, setActiveSession] = useState("new");
  const [sessionId, setSessionId] = useState("session-initial");
  const [caseQuery, setCaseQuery] = useState("");
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionsHydrated, setSessionsHydrated] = useState(false);
  const [deviceCatalog, setDeviceCatalog] = useState<DeviceProfile[]>(fallbackDeviceCatalog);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<DeviceCategory>("router");
  const [selectedModel, setSelectedModel] = useState("Archer C6");
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [sessionMenuOpen, setSessionMenuOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [navigationOpen, setNavigationOpen] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const requestController = useRef<AbortController | null>(null);
  const messageSequence = useRef(0);
  const voiceClient = useRef<FridayVoiceClient | null>(null);
  const voiceAssistantId = useRef<string | null>(null);
  const voiceTurnId = useRef<string | null>(null);
  const sessionStarted = useRef(false);
  const composerInput = useRef<HTMLTextAreaElement | null>(null);
  const [voiceConnected, setVoiceConnected] = useState(false);
  const [voiceCaptureEnabled, setVoiceCaptureEnabled] = useState(true);
  const threadEnd = useRef<HTMLDivElement | null>(null);

  function resizeComposer() {
    const input = composerInput.current;
    if (!input) return;
    input.style.height = "0px";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  }

  function createMessageId(role: "user" | "assistant"): string {
    messageSequence.current += 1;
    // A timestamp alone can collide when a final voice transcript creates the
    // user and assistant messages in the same render. Use a UUID so IDs remain
    // stable even across Next.js Fast Refresh boundaries.
    const uniquePart = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${messageSequence.current}`;
    return `${role}-${uniquePart}`;
  }

  const selectedDevice = deviceCatalog.find((device) => device.category === selectedCategory && device.name === selectedModel) ?? deviceCatalog[0];
  const devicesInCategory = deviceCatalog.filter((device) => device.category === selectedCategory);

  function startNewSession(category = selectedCategory, model?: string) {
    requestController.current?.abort();
    void stopVoice();
    setSelectedCategory(category);
    setSelectedModel(model ?? deviceCatalog.find((device) => device.category === category)?.name ?? selectedModel);
    setActiveSession("new");
    sessionStarted.current = false;
    const nextSessionId = `session-${globalThis.crypto?.randomUUID?.() ?? Date.now()}`;
    setSessionId(nextSessionId);
    setCaseQuery("");
    setMessages([]);
    setSelectedAnswer(null);
    setState("ready");
    setApiError(null);
    setEvidenceOpen(false);
    setNavigationOpen(false);
    setSessionMenuOpen(false);
  }

  function chooseSession(session: Session) {
    requestController.current?.abort();
    void stopVoice();
    setActiveSession(session.id);
    sessionStarted.current = true;
    setSessionId(session.id);
    setCaseQuery(session.title);
    setSelectedCategory(session.category);
    setSelectedModel(session.device);
    setMessages(session.messages);
    setSelectedAnswer(session.selectedAnswer);
    setState("ready");
    setApiError(null);
    setEvidenceOpen(false);
    setNavigationOpen(false);
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

  function ensureSessionStarted(title: string) {
    if (sessionStarted.current) return;
    sessionStarted.current = true;
    setCaseQuery(title);
    setActiveSession(sessionId);
    const timestamp = new Date().toISOString();
    setSessions((current) => current.some((session) => session.id === sessionId)
      ? current
      : [{
          id: sessionId,
          title,
          device: selectedDevice.name,
          category: selectedCategory,
          status: "active",
          createdAt: timestamp,
          updatedAt: timestamp,
          messages: [],
          selectedAnswer: null,
        }, ...current]);
  }

  async function runTroubleshoot(
    query: string,
    displayText = query,
    addUser = true,
    interaction: { observation?: string; selectedOption?: string; regenerate?: boolean } = {},
  ) {
    requestController.current?.abort();
    // Typed answers and option choices must be able to interrupt a spoken
    // answer without closing the persistent microphone session.
    voiceClient.current?.cancelAssistant();
    const controller = new AbortController();
    requestController.current = controller;
    setApiError(null);
    setState("thinking");
    if (addUser) {
      setMessages((current) => [...current, { id: createMessageId("user"), role: "user", text: displayText }]);
      ensureSessionStarted(displayText);
    }

    const manufacturer = selectedDevice.manufacturer;
    try {
      const assistantId = createMessageId("assistant");
      setMessages((current) => [...current, { id: assistantId, role: "assistant", text: "" }]);
      let completed: TroubleshootingResponse | null = null;
      await troubleshootStream(
        {
          query,
          manufacturer,
          model: selectedDevice.name,
          session_id: sessionId,
          observation: interaction.observation,
          selected_option: interaction.selectedOption,
          regenerate: interaction.regenerate,
        },
        (event) => {
          if (event.type === "token") {
            setMessages((current) => current.map((message) => (
              message.id === assistantId ? { ...message, text: `${message.text}${event.text}` } : message
            )));
          }
          if (event.type === "complete") {
            completed = event.response;
            const result = event.response;
            const answerText = responseText(result);
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
    void runTroubleshoot(text, text, true, { observation: text });
    setDraft("");
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }

  async function copyMessage(id: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      window.setTimeout(() => setCopiedId((current) => (current === id ? null : current)), 1600);
    } catch {
      setApiError("Copy is not available in this browser context.");
    }
  }

  function regenerateLatest() {
    const latestUserMessage = [...messages].reverse().find((message) => message.role === "user");
    if (!latestUserMessage || state === "thinking") return;
    // Regenerate reuses the same session without recording a duplicate user turn.
    void runTroubleshoot(latestUserMessage.text, "", false, { regenerate: true });
  }

  function completeAssistant(id: string, response: TroubleshootingResponse) {
    const answerText = responseText(response);
    setMessages((current) => current.map((message) => (message.id === id ? { ...message, text: answerText, response } : message)));
  }

  function handleVoiceEvent(event: VoiceEvent) {
    if (event.type === "voice.connecting" || event.type === "voice.ready") {
      setState("connecting");
      return;
    }
    if (event.type === "session.ready") {
      setApiError(null);
      setVoiceConnected(true);
      setState("listening");
      return;
    }
    if (event.type === "speech.start") {
      requestController.current?.abort();
      // If Friday was speaking, this partial is an explicit interruption.
      setState((current) => (current === "speaking" ? "interrupting" : "listening"));
      // Fall through to listening on next partial; interrupting is transient.
      window.setTimeout(() => setState("listening"), 300);
      return;
    }
    if (event.type === "speech.end") {
      setState("thinking");
      return;
    }
    if (event.type === "tool") {
      setState("thinking");
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
      voiceTurnId.current = event.turn_id;
      setDraft("");
      setSelectedAnswer(null);
      setApiError(null);
      setMessages((current) => [
        ...current,
        { id: createMessageId("user"), role: "user", text: event.text },
        { id: assistantId, role: "assistant", text: "" },
      ]);
      ensureSessionStarted(event.text);
      setState("thinking");
      return;
    }
    if (event.type === "assistant.token" && voiceAssistantId.current && event.turn_id === voiceTurnId.current) {
      setMessages((current) => current.map((message) => (
        message.id === voiceAssistantId.current ? { ...message, text: `${message.text}${event.text}` } : message
      )));
      return;
    }
    if (
      event.type === "assistant.complete" &&
      voiceAssistantId.current &&
      event.turn_id === voiceTurnId.current
    ) {
      completeAssistant(voiceAssistantId.current, event.response);
      setState(event.response.status === "ready" ? "speaking" : "listening");
      return;
    }
    if (event.type === "assistant.audio_complete" && event.turn_id === voiceTurnId.current) {
      setState("listening");
      return;
    }
    if (event.type === "assistant.cancelled") {
      if (event.turn_id && event.turn_id !== voiceTurnId.current) return;
      const id = voiceAssistantId.current;
      if (id) setMessages((current) => current.filter((message) => message.id !== id || Boolean(message.response)));
      voiceAssistantId.current = null;
      voiceTurnId.current = null;
      setState("listening");
      return;
    }
    if (event.type === "voice.error") {
      setVoiceConnected(false);
      voiceClient.current = null;
      setApiError(event.message);
      setState("error");
      return;
    }
    if (event.type === "voice.closed") {
      voiceClient.current = null;
      setVoiceConnected(false);
      setApiError(event.message);
      setState("recovering");
    }
  }

  async function startVoice() {
    try {
      if (voiceClient.current) return;
      const client = new FridayVoiceClient(handleVoiceEvent);
      voiceClient.current = client;
      setVoiceCaptureEnabled(true);
      setState("connecting");
      await client.start({ sessionId, manufacturer: selectedDevice.manufacturer, model: selectedDevice.name });
      setApiError(null);
      // session.ready is the authoritative point at which the server has
      // accepted the device scope and can receive microphone audio.
    } catch (error) {
      voiceClient.current = null;
      setVoiceConnected(false);
      setApiError(error instanceof Error ? error.message : "Could not start the microphone.");
      setState("interrupted");
    }
  }

  async function stopVoice() {
    const client = voiceClient.current;
    voiceClient.current = null;
    await client?.stop();
    setVoiceConnected(false);
    setVoiceCaptureEnabled(false);
    setState("ready");
  }

  function toggleVoiceCapture() {
    const enabled = !voiceCaptureEnabled;
    voiceClient.current?.setCaptureEnabled(enabled);
    setVoiceCaptureEnabled(enabled);
    if (!enabled) setDraft("");
  }

  function interruptVoice() {
    voiceClient.current?.cancelAssistant();
    setState("listening");
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
    const controller = new AbortController();
    void getSupportedDevices(controller.signal)
      .then((devices) => {
        if (devices.length > 0) {
          setDeviceCatalog(devices.map(deviceProfile));
          setCatalogError(null);
        }
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setCatalogError("The supported-device catalog is unavailable. Troubleshooting remains limited to the current device.");
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!sessionsHydrated) return;
    window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessions));
  }, [sessions, sessionsHydrated]);

  useEffect(() => {
    resizeComposer();
  }, [draft]);

  useEffect(() => {
    if (messages.length === 0) return;
    const container = threadEnd.current?.parentElement;
    if (container) container.scrollTop = container.scrollHeight;
  }, [messages]);

  useEffect(() => {
    if (!navigationOpen && !evidenceOpen) return;
    const panel = document.querySelector<HTMLElement>(navigationOpen ? ".mobile-navigation" : ".diagnostic-rail");
    const previous = document.activeElement as HTMLElement | null;
    const controls = () => Array.from(panel?.querySelectorAll<HTMLElement>('button:not([disabled]), a[href], select, summary') ?? []);
    controls()[0]?.focus();
    const keydown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") { setNavigationOpen(false); setEvidenceOpen(false); }
      if (event.key === "Tab") {
        const items = controls(); const first = items[0]; const last = items.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener("keydown", keydown);
    return () => { document.removeEventListener("keydown", keydown); previous?.focus(); };
  }, [navigationOpen, evidenceOpen]);

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
  const latestAssistantMessage = [...messages].reverse().find((message) => message.role === "assistant" && message.response);
  const latestResponse = latestAssistantMessage?.response;
  const latestCitation = latestResponse?.citations[0];
  const activeQuestion = latestResponse?.status === "ready"
    ? latestResponse.turn?.observation_request?.question ?? latestResponse.step?.question
    : undefined;
  const observations = latestResponse?.observations ?? [];
  const confirmedFacts = Object.values(latestResponse?.facts ?? {});
  const factTransitions = Object.values(latestResponse?.fact_history ?? {}).flatMap((history) => {
    const latest = history.at(-1);
    if (!latest?.previous_value || latest.previous_value === latest.value) return [];
    return [`${latest.label}: ${latest.previous_value} → ${latest.value}`];
  });
  const hasReportedProblem = messages.some((message) => message.role === "user");
  const hasEvidence = Boolean(latestCitation);
  const hasConfirmedState = confirmedFacts.length > 0 || observations.length > 0;
  const isWaitingForObservation = Boolean(activeQuestion) && !isThinking;
  const orderedSessions = [...sessions].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));

  return (
    <main className={`app-shell ${messages.length === 0 ? "session-is-empty" : "session-has-messages"} ${voiceConnected || state === "connecting" ? "voice-active" : ""}`}>
      <a className="skip-link" href="#conversation">Skip to conversation</a>
      <header className="topbar">
        <Brand/>
        <div className="topbar-center"><span className="topbar-context">Workspace<span>/</span>{caseQuery ? "Troubleshooting session" : "New session"}</span></div>
        <div className="topbar-actions">
          <a href="/">About Friday <ArrowTopRightOnSquareIcon/></a>
          <button className="mobile-nav-toggle" type="button" aria-label="Open sessions and devices" aria-expanded={navigationOpen} onClick={() => setNavigationOpen((open) => !open)}><Bars3Icon/></button>
        </div>
      </header>

      <div className="workspace">
        <aside className="session-sidebar" aria-label="Device and troubleshooting sessions">
          <button className="new-session-quiet" type="button" onClick={() => startNewSession()}><PlusIcon/> New session</button>
          <div className="device-picker">
            <span className="sidebar-title">Your device</span>
            {Object.entries(deviceCategories).map(([category, device]) => (
              <button className={`device-category ${selectedCategory === category ? "selected" : ""}`} key={category} type="button" onClick={() => startNewSession(category as DeviceCategory)}>
                <Icon name={device.icon} /><span>{device.label}</span>
              </button>
            ))}
          </div>

          <div className="current-device">
            <span className="sidebar-title">Model</span>
            <div className="current-device-card"><span className="current-device-image"><Icon name={selectedDevice.icon} /></span><div><strong>{selectedDevice.manufacturer} {selectedDevice.name}</strong><span>{selectedDevice.detail}</span><label className="device-model-select"><span className="sr-only">Select a supported device model</span><select value={selectedDevice.name} onChange={(event) => startNewSession(selectedCategory, event.target.value)}>{devicesInCategory.map((device) => <option key={`${device.manufacturer}-${device.name}`} value={device.name}>{device.manufacturer} {device.name}</option>)}</select></label></div></div>
            {catalogError && <p className="catalog-error" role="status">{catalogError}</p>}
          </div>

          <div className="sidebar-heading"><h2>Recent sessions</h2><span>{orderedSessions.length}</span></div>
          <div className="session-list">
            {orderedSessions.map((session) => (
              <button className={`session-item ${activeSession === session.id ? "selected" : ""}`} key={session.id} type="button" onClick={() => chooseSession(session)}>
                <span className={`session-status-dot ${activeSession === session.id ? "active" : session.status}`} aria-hidden="true" />
                <span className="session-item-copy"><strong>{session.title}</strong><span>{session.device}</span></span>
                <span className="session-time">{sessionStatusLabel(session.status)}<small>{relativeSessionTime(session.updatedAt)}</small></span>
              </button>
            ))}
            {sessionsHydrated && orderedSessions.length === 0 && <p className="session-empty">Your troubleshooting history will appear here.</p>}
          </div>
          <p className="local-sessions-note"><ComputerDesktopIcon/><span>Saved on this browser<small>Your sessions stay here until deleted.</small></span></p>
        </aside>

        {navigationOpen && <button className="mobile-nav-backdrop" type="button" aria-label="Close case navigation" onClick={() => setNavigationOpen(false)} />}
        <aside className={`mobile-navigation ${navigationOpen ? "mobile-navigation-open" : ""}`} inert={!navigationOpen} role="dialog" aria-modal={navigationOpen || undefined} aria-label="Mobile device and case navigation">
          <div className="mobile-navigation-header"><Brand/><button type="button" aria-label="Close navigation" onClick={() => setNavigationOpen(false)}><XMarkIcon/></button></div>
          <div className="mobile-navigation-section"><span className="sidebar-title">DEVICE</span>{Object.entries(deviceCategories).map(([category, device]) => <button className={`device-category ${selectedCategory === category ? "selected" : ""}`} key={category} type="button" onClick={() => startNewSession(category as DeviceCategory)}><Icon name={device.icon} /><span>{device.label}</span></button>)}</div>
          <div className="mobile-navigation-section"><label className="sidebar-title" htmlFor="mobile-device-model">Model</label><select id="mobile-device-model" className="mobile-model-select" value={selectedDevice.name} onChange={(event) => startNewSession(selectedCategory, event.target.value)}>{devicesInCategory.map((device) => <option key={`${device.manufacturer}-${device.name}`} value={device.name}>{device.manufacturer} {device.name}</option>)}</select></div>
          <button className="new-session-quiet" type="button" onClick={() => startNewSession()}><PlusIcon/> New session</button>
          <div className="mobile-navigation-section"><span className="sidebar-title">CASES</span>{orderedSessions.map((session) => <button className={`mobile-case ${activeSession === session.id ? "selected" : ""}`} key={session.id} type="button" onClick={() => chooseSession(session)}><strong>{session.title}</strong><span>{sessionStatusLabel(session.status)} · {session.device}</span></button>)}{sessionsHydrated && orderedSessions.length === 0 && <p className="session-empty">No saved cases yet.</p>}</div>
        </aside>

        <section className="conversation" id="conversation" aria-labelledby="conversation-title">
          <div className="troubleshooting-thread">
            <div className="conversation-header">
              <div><span className="case-context"><Icon name={selectedDevice.icon}/>{selectedDevice.manufacturer} {selectedDevice.name}</span><h1 id="conversation-title">{caseQuery || "Let’s figure it out."}</h1></div>
              <div className="session-menu-wrap">
                <button className="session-menu-button" type="button" aria-label="Session actions" aria-expanded={sessionMenuOpen} onClick={() => setSessionMenuOpen((open) => !open)}><Icon name="more" /></button>
                {sessionMenuOpen && <div className="session-menu" role="menu"><button type="button" onClick={() => startNewSession()}>Start a new session</button>{activeSession !== "new" && <button type="button" onClick={() => void deleteCurrentSession()}>Delete this session</button>}</div>}
              </div>
              <button className="evidence-toggle" type="button" aria-expanded={evidenceOpen} onClick={() => setEvidenceOpen((open) => !open)}><BookOpenIcon/> Evidence</button>
            </div>

            {hasReportedProblem && <div className="diagnostic-progress" aria-label="Session status"><span><i/>{isThinking ? "Working on your response" : isWaitingForObservation ? "Waiting for your observation" : "Conversation in progress"}</span>{hasEvidence && <span><BookOpenIcon/>{latestResponse?.citations.length} source{latestResponse?.citations.length === 1 ? "" : "s"} available</span>}</div>}
            <div className="message-list" aria-live="polite">
              {messages.length === 0 && <div className="empty-thread">
                <div className="empty-device-art" aria-hidden="true"><span/><span/><div><Icon name={selectedDevice.icon}/></div></div>
                <h2>What isn’t working?</h2>
                <p>Describe what you’re seeing. We’ll work through it with your device’s manual close at hand.</p>
                <div className="starter-prompts" aria-label="Start with an example">{(selectedCategory === "router" ? ["Wi-Fi connects, but there’s no internet", "Help me understand the router lights", "The connection keeps dropping"] : selectedCategory === "printer" ? ["My printer is on but won’t print", "There’s an error on the display", "The paper keeps getting stuck"] : ["My laptop isn’t charging", "It won’t start up", "I’m having trouble connecting to Wi-Fi"]).map(prompt => <button type="button" key={prompt} onClick={() => { setDraft(prompt); composerInput.current?.focus(); }}>{prompt}<ArrowRightIcon/></button>)}</div>
                {!voiceConnected && state !== "connecting" && <div className="voice-entry"><button className="empty-voice-link" type="button" onClick={() => void startVoice()}><MicrophoneIcon/><span>Talk to Friday</span><span className="voice-entry-wave" aria-hidden="true"><i/><i/><i/><i/><i/></span></button><small>Keep your hands on the device. Tell us what you see.</small></div>}
              </div>}
              {messages.filter((message) => message.role !== "assistant" || message.text || message.response).map((message, index) => {
                const response = message.response;
                const isLatestResponse = message.id === latestAssistantMessage?.id;
                const selectedHistoricalAnswer = messages.slice(index + 1).find((item) => item.role === "user")?.text;
                const options = response?.turn?.observation_request?.options ?? response?.step?.options ?? [];
                return (
                  <article className={`message message-row diagnostic-entry ${message.role}`} key={message.id}>
                    {message.role === "user" && <div className="chat-bubble user-bubble"><p>{message.text}</p></div>}
                    {message.role === "assistant" && !response && message.text && <div className="assistant-message streaming-response"><div className="assistant-avatar"><BrandMark/></div><div className="chat-bubble assistant-bubble"><p className="response-copy">{message.text}<span className="stream-caret"/></p></div></div>}
                    {message.role === "assistant" && response && (
                      <div className={`assistant-message ${isLatestResponse ? "active-response" : "history-response"}`}>
                        <div className="assistant-avatar" aria-hidden="true"><BrandMark/></div>
                        <div className="chat-bubble assistant-bubble">
                          <p className="response-copy">{message.text}</p>
                        {response.status === "abstained" && response.missing_observations.length > 0 && <ul className="missing-observations">{response.missing_observations.map((observation) => <li key={observation}>{observation}</li>)}</ul>}
                        {options.length > 0 && <div className="answer-options" aria-label="Diagnostic answer options">{options.map((option) => {
                          const isSelected = isLatestResponse ? selectedAnswer === option.label : selectedHistoricalAnswer === option.label;
                          return <button className={isSelected ? "selected" : ""} key={option.id} type="button" aria-pressed={isSelected} disabled={!isLatestResponse} onClick={() => submitAnswer(option)}>{option.label}</button>;
                        })}</div>}
                        {response.images.length > 0 && <div className="manual-images" aria-label="Figures from the manufacturer manual">{response.images.map((image) => <figure key={image.asset_id}><img src={`${API_BASE_URL}${image.url}`} alt={`${image.document_title}, page ${image.page}`} /><figcaption>{image.document_title} · p. {image.page}</figcaption></figure>)}</div>}
                        {response.citations[0] && <div className="source-line"><Icon name="manual" /><a href={response.citations[0].source_url || "#source"} target="_blank" rel="noreferrer">{response.citations[0].document_title} · p. {response.citations[0].page} · {response.citations[0].section}</a><Icon name="external" /></div>}
                          <div className="response-actions">
                            <button
                              type="button"
                              onClick={() => void copyMessage(message.id, message.text)}
                              aria-label={copiedId === message.id ? "Response copied" : "Copy response"}
                            >
                              {copiedId === message.id ? <CheckIcon/> : <DocumentDuplicateIcon/>}<span>{copiedId === message.id ? "Copied" : "Copy"}</span>
                            </button>
                            {isLatestResponse && <button
                              type="button"
                              onClick={regenerateLatest}
                              disabled={state === "thinking"}
                            >
                              <ArrowPathIcon/><span>Try again</span>
                            </button>}
                          </div>
                        </div>
                      </div>
                    )}
                  </article>
                );
              })}
              {apiError && <div className="api-error" role="alert"><strong>Couldn&apos;t check the manuals.</strong><span>{apiError}</span><button type="button" onClick={() => { const latestUserMessage = [...messages].reverse().find((message) => message.role === "user"); if (latestUserMessage) void runTroubleshoot(latestUserMessage.text, "", false); }}>Try again</button></div>}
              {isThinking && !messages.at(-1)?.text && <div className="thinking-line" role="status"><BrandMark/><span className="thinking-pulse" /> Working on your response</div>}
              <div ref={threadEnd} />
            </div>

            <div className="composer-wrap">
              {(voiceConnected || state === "connecting") ? <section className={`voice-console voice-state-${!voiceCaptureEnabled && state === "listening" ? "paused" : state} ${isListening ? "voice-console-listening" : ""}`} aria-label="Voice controls">
                <div className="voice-console-header">
                  <div className="voice-console-title"><BrandMark/><span>Friday voice</span></div>
                  <button className="voice-end-button" type="button" onClick={() => void stopVoice()}><XMarkIcon/>End voice</button>
                </div>
                <div className="voice-console-body">
                  <div className="voice-signal" aria-hidden="true"><svg viewBox="0 0 240 96" fill="none">{Array.from({ length: 35 }, (_, index) => <line key={index} x1={18 + index * 6} x2={18 + index * 6} y1={48 - (8 + Math.sin(index * .72) ** 2 * (32 - Math.abs(index - 17)))} y2={48 + (8 + Math.sin(index * .72) ** 2 * (32 - Math.abs(index - 17)))} style={{ animationDelay: `${index * -63}ms` }}/>)}</svg></div>
                  <div className="voice-transcript-wrap"><strong className="voice-state-label" role="status">{state === "connecting" ? "Getting connected…" : state === "speaking" ? "Friday is speaking" : state === "thinking" ? "Putting your answer together" : voiceCaptureEnabled ? "I’m listening." : "Microphone paused"}</strong><p className="voice-transcript">{draft || (state === "speaking" ? [...messages].reverse().find(message => message.role === "assistant")?.text || "Your reply will appear here as it arrives." : state === "thinking" ? "Your observation is in. The reply will appear in the conversation." : !voiceCaptureEnabled ? "Take your time. Resume whenever you’re ready." : "Describe the problem, or tell me what changed.")}</p></div>
                </div>
                <div className="voice-console-footer">
                  <button className={`voice-control-button ${voiceCaptureEnabled ? "selected" : ""}`} type="button" disabled={state === "connecting"} onClick={toggleVoiceCapture}><Icon name={voiceCaptureEnabled ? "pause" : "mic"} /><span>{voiceCaptureEnabled ? "Pause microphone" : "Resume microphone"}</span></button>
                  {state === "speaking" && <button className="voice-control-button interrupt-control" type="button" onClick={interruptVoice}><Icon name="stop" /><span>Interrupt Friday</span></button>}
                  <button className="voice-to-text" type="button" onClick={() => void stopVoice()}>Switch to typing</button>
                </div>
              </section> : <form className="composer" onSubmit={submitMessage}>
                <label className="sr-only" htmlFor="message">Describe what you see</label>
                <textarea ref={composerInput} id="message" rows={1} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder={messages.length === 0 ? "Tell Friday what’s happening…" : "What did you notice?"} />
                <div className="composer-toolbar"><span className="composer-device"><Icon name={selectedDevice.icon}/>{selectedDevice.name}</span><button className="mic-button" type="button" aria-label="Start voice input" onClick={() => void startVoice()}><Icon name="mic" /><span>Talk to Friday</span></button><button className="send-button visible" type="submit" aria-label="Send observation" disabled={!draft.trim()}><Icon name="arrow" /></button></div>
              </form>}
              <p className="composer-note">Friday can make mistakes. Check the linked manual before acting.</p>
            </div>
          </div>
        </section>

        {evidenceOpen && <button className="evidence-backdrop" type="button" aria-label="Close evidence panel" onClick={() => setEvidenceOpen(false)}/>}
        <aside className={`diagnostic-rail ${evidenceOpen ? "mobile-open" : ""}`} aria-label="Diagnostic state and evidence">
          <div className="rail-header"><div><BookOpenIcon/><h2>Evidence & context</h2></div><button className="rail-toggle" type="button" aria-label="Close diagnostic state" onClick={() => setEvidenceOpen(false)}><XMarkIcon/></button></div>
          <div className="rail-section"><div className="rail-label">CURRENT DEVICE</div><p className="rail-device">{selectedDevice.manufacturer} {selectedDevice.name}<span>{selectedDevice.detail}</span></p></div>
          <div className="rail-section"><div className="rail-label">CONFIRMED</div>{confirmedFacts.length > 0 ? <dl className="fact-list">{confirmedFacts.map((fact) => <div className="fact-row" key={fact.key}><dt>{factKeyLabel(fact.key)}</dt><dd><span>{fact.value}</span><Icon name="check" /></dd></div>)}</dl> : observations.length > 0 ? <ul className="observation-list">{observations.map((observation) => <li key={observation}><span className="observation-dot done" /><span>{observation}</span></li>)}</ul> : <p className="rail-empty">No confirmed observations yet.</p>}{factTransitions.length > 0 && <div className="fact-transitions"><div className="rail-label">CHANGED AFTER CHECK</div>{factTransitions.map((transition) => <p key={transition}>{transition}</p>)}</div>}</div>
          {(activeQuestion || (latestResponse?.missing_observations ?? []).length > 0) && <div className="rail-section"><div className="rail-label">Next to establish</div><ul className="unknown-list">{activeQuestion && <li><span>{activeQuestion}</span></li>}{!activeQuestion && latestResponse?.missing_observations.map((observation) => <li key={observation}><span>{observation}</span></li>)}</ul></div>}
          {!latestCitation && <div className="rail-source-empty"><BookOpenIcon/><strong>The source stays with the answer.</strong><p>Relevant manual pages and references will appear here as you troubleshoot.</p></div>}
          {latestCitation && <div className="rail-section evidence-section"><div className="rail-label">SOURCE</div><div className="evidence-card"><strong>{latestCitation.document_title}</strong><span>{latestCitation.section}</span><span className="evidence-page">Page {latestCitation.page}</span><details className="manual-viewer"><summary>View original page</summary><div className="manual-viewer-content"><iframe title={`${latestCitation.document_title}, page ${latestCitation.page}`} src={`${latestCitation.source_url || "about:blank"}#page=${latestCitation.page}`} loading="lazy" /></div></details><a href={latestCitation.source_url || "#source"} target="_blank" rel="noreferrer">Open manual page <Icon name="arrow" /></a></div></div>}
        </aside>
      </div>
    </main>
  );
}
