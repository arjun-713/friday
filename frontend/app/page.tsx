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
import { API_BASE_URL, troubleshoot, TroubleshootingApiError, type DiagnosticOption, type TroubleshootingResponse } from "../lib/api";

type Message = { id: number; role: "user" | "assistant"; text: string; meta?: string; response?: TroubleshootingResponse };
type SessionState = "ready" | "listening" | "thinking" | "speaking" | "interrupted";
type DeviceCategory = "laptop" | "router" | "printer";
type SessionStatus = "active" | "open" | "resolved";

type Session = {
  id: string;
  title: string;
  device: string;
  time: string;
  status: SessionStatus;
};

const sessions: Session[] = [
  { id: "router-01", title: "Wi-Fi, no internet", device: "TP-Link Archer C6", time: "Now", status: "active" },
  { id: "printer-02", title: "Printer not responding", device: "Brother HL-L2350DW", time: "Yesterday", status: "open" },
  { id: "laptop-03", title: "Laptop won’t charge", device: "ThinkPad T480", time: "Aug 12", status: "resolved" },
];

const deviceProfiles = {
  laptop: { name: "ThinkPad T480", detail: "Laptop / Desktop", icon: "laptop" as const },
  router: { name: "TP-Link Archer C6", detail: "Wi-Fi Router", icon: "router" as const },
  printer: { name: "HP LaserJet Pro M428fdw", detail: "Printer", icon: "printer" as const },
};

const initialMessages: Message[] = [
  { id: 1, role: "user", text: "My router is on, but nothing can get online. The Wi-Fi name still shows up.", meta: "just now" },
];

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

function ManualFigure({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`manual-figure ${compact ? "compact" : ""}`} aria-label="Manual figure showing router status lights">
      <svg width="240" height="150" viewBox="0 0 240 150" fill="none" role="img" aria-hidden="true">
        <path d="M25 112h190" className="figure-line" /><path d="M45 103V61l72-24 78 24v42" className="figure-outline" /><path d="M45 61h150M74 53v50M104 43v60M134 48v55M164 55v48" className="figure-line" /><rect x="64" y="77" width="104" height="17" rx="3" className="figure-panel" /><circle cx="82" cy="85" r="4" className="figure-light" /><circle cx="103" cy="85" r="4" className="figure-light active" /><circle cx="124" cy="85" r="4" className="figure-light" /><path d="m112 132 20-17m-20 17 3-25m-3 25-18-9" className="figure-arrow" />
      </svg>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [draft, setDraft] = useState("");
  const [state, setState] = useState<SessionState>("ready");
  const [activeSession, setActiveSession] = useState("router-01");
  const [sessionId, setSessionId] = useState("router-01");
  const [caseQuery, setCaseQuery] = useState(initialMessages[0].text);
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

  const selectedDevice = deviceProfiles[selectedCategory];

  function startNewSession(category = selectedCategory) {
    requestController.current?.abort();
    setSelectedCategory(category);
    setActiveSession("new");
    const nextSessionId = `session-${Date.now()}`;
    setSessionId(nextSessionId);
    setCaseQuery("");
    setMessages([]);
    setSelectedAnswer(null);
    setState("ready");
    setApiError(null);
  }

  function chooseSession(session: Session) {
    requestController.current?.abort();
    setActiveSession(session.id);
    setSessionId(session.id);
    setCaseQuery(session.title);
    setSelectedCategory(session.id === "printer-02" ? "printer" : session.id === "laptop-03" ? "laptop" : "router");
    setMessages(session.id === "router-01" ? initialMessages : []);
    setSelectedAnswer(null);
    setState("ready");
    setApiError(null);
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
    if (addUser) setMessages((current) => [...current, { id: Date.now(), role: "user", text: displayText, meta: "just now" }]);

    const manufacturer = selectedCategory === "router" ? "TP-Link" : selectedCategory === "printer" ? "HP" : "Lenovo";
    try {
      const result = await troubleshoot(
        {
          query,
          manufacturer,
          model: selectedDevice.name,
          session_id: sessionId,
          observation: interaction.observation,
          selected_option: interaction.selectedOption,
        },
        controller.signal,
      );
      const answerText = result.status === "ready" ? result.step?.instruction ?? result.answer ?? "The manual does not provide an answer for this observation." : result.missing_observations.length > 0 ? `I need one more observation: ${result.missing_observations.join(", ")}.` : "I could not verify a safe next step from the available manuals.";
      setMessages((current) => [...current, { id: Date.now() + 1, role: "assistant", text: answerText, meta: "just now", response: result }]);
      setState("ready");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setApiError(error instanceof TroubleshootingApiError ? error.message : "The troubleshooting service could not be reached.");
      setState("interrupted");
    }
  }

  function submitAnswer(option: DiagnosticOption) {
    setSelectedAnswer(option.label);
    void runTroubleshoot(
      `${caseQuery || selectedDevice.name} Observation: ${option.label}`,
      option.label,
      true,
      { selectedOption: option.id },
    );
  }

  function submitMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = draft.trim();
    if (!text) return;
    if (!caseQuery) setCaseQuery(text);
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

  useEffect(() => {
    void runTroubleshoot(initialMessages[0].text, "", false);
    return () => requestController.current?.abort();
  }, []);

  const isListening = state === "listening";
  const isThinking = state === "thinking";

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="wordmark" href="#conversation" aria-label="Friday home"><span className="wordmark-mark"><Icon name="router" /></span><span>friday</span></a>
        <div className="topbar-center"><span className="topbar-context">TP-Link Archer C6 <span>/</span> Wi-Fi visible, no internet</span></div>
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
            {sessions.map((session) => (
              <button className={`session-item ${activeSession === session.id ? "selected" : ""}`} key={session.id} type="button" onClick={() => chooseSession(session)}>
                <span className={`session-status-dot ${session.status}`} aria-hidden="true" />
                <span className="session-item-copy"><strong>{session.title}</strong><span>{session.device}</span></span>
                <span className="session-time">{session.status === "resolved" ? "Resolved" : session.time}</span>
              </button>
            ))}
          </div>
          <button className="view-sessions" type="button">View all sessions <Icon name="arrow" /></button>
          <button className="profile-row" type="button" aria-label="Open profile"><span className="profile-avatar"><Icon name="user" /></span><span className="profile-copy"><strong>Profile</strong><span>Account settings</span></span><Icon name="chevron" /></button>
        </aside>

        <section className="conversation" id="conversation" aria-labelledby="conversation-title">
          <div className="troubleshooting-thread">
            <div className="conversation-header">
              <div><span className="case-context">WI-FI ROUTER / TP-LINK ARCHER C6</span><h1 id="conversation-title">Wi-Fi visible, no internet</h1></div>
              <div className="session-menu-wrap">
                <button className="session-menu-button" type="button" aria-label="Session actions" aria-expanded={sessionMenuOpen} onClick={() => setSessionMenuOpen((open) => !open)}><Icon name="more" /></button>
                {sessionMenuOpen && <div className="session-menu" role="menu"><button type="button">Rename session</button><button type="button" onClick={() => setMessages([])}>Start over</button><button type="button" onClick={() => setMessages([])}>Clear conversation</button></div>}
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
                    <div className={`step-panel ${message.response?.status === "abstained" ? "abstained-panel" : ""}`}>
                      <div className="step-heading"><h2>{message.response?.status === "abstained" ? "I need another observation" : message.response?.step?.title ?? "Next check"}</h2></div>
                      <p className="instruction response-copy">{message.text}</p>
                      {message.response?.status === "abstained" ? <ul className="missing-observations">{message.response.missing_observations.map((observation) => <li key={observation}>{observation}</li>)}</ul> : <>
                        {message.response?.step && <div className="procedure-content"><div className="procedure-copy"><p className="instruction">{message.response.step.question}</p></div>{selectedCategory === "router" && <ManualFigure />}</div>}
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
                {isListening ? <div className="waveform" aria-live="polite"><span>Listening</span><i /><i /><i /><i /><i /></div> : <input id="message" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Describe what you see…" />}
                <input className="sr-only" id="attachment" type="file" accept="image/*,.pdf,.txt" onChange={handleAttachment} />
                <label className="attach-button" htmlFor="attachment"><Icon name="paperclip" /><span>Attach</span></label>
                {isListening ? <button className="mic-button active" type="button" aria-label="Stop listening" onClick={() => setState("interrupted")}><Icon name="pause" /></button> : <button className={`mic-button ${draft ? "quiet" : "primary"}`} type="button" aria-label="Start voice input" onClick={() => setState("listening")}><Icon name="mic" /></button>}
                {!isListening && <button className={`send-button ${draft ? "visible" : ""}`} type="submit" aria-label="Send observation" disabled={!draft.trim()}><Icon name="send" /></button>}
              </form>
            </div>
          </div>
        </section>

        <aside className={`diagnostic-rail ${evidenceOpen ? "mobile-open" : ""}`} aria-label="Evidence ledger">
          <div className="rail-header"><h2>What we know</h2><button className="rail-toggle" type="button" aria-label="Collapse evidence ledger"><Icon name="chevron" /></button></div>
          <div className="rail-section"><div className="rail-label">OBSERVED</div><ul className="observation-list"><li><span className="observation-dot done" /> <span>Router has power</span></li><li><span className="observation-dot done" /> <span>Wi-Fi network is visible</span></li></ul></div>
          <div className="rail-section"><div className="rail-label">{selectedAnswer ? "OBSERVED" : "NEED TO VERIFY"}</div><ul className="observation-list"><li className={selectedAnswer ? "observation-known" : ""}><span className={`observation-dot ${selectedAnswer ? "done" : "pending"}`} /> <span>{selectedAnswer ? `WAN light ${selectedAnswer.toLowerCase()}` : "WAN light state"}</span></li></ul></div>
          <div className="rail-section evidence-section"><div className="rail-label">MANUAL EVIDENCE</div><div className="evidence-card"><strong>Archer C6 User Guide</strong><span>Page 42 · LED descriptions</span><ManualFigure compact /><a href="#source">Open page 42 <Icon name="arrow" /></a></div></div>
        </aside>
      </div>
    </main>
  );
}
