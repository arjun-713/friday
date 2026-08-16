"use client";

import { ChangeEvent, FormEvent, useState } from "react";

type Message = { id: number; role: "user" | "assistant"; text: string; meta?: string };
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
  { id: 2, role: "assistant", text: "Check the WAN light.", meta: "Friday" },
];

function Icon({ name }: { name: "arrow" | "mic" | "send" | "check" | "pause" | "chevron" | "laptop" | "router" | "printer" | "external" | "manual" | "more" | "paperclip" | "user" }) {
  const paths = {
    arrow: "M4 12h15m0 0-5-5m5 5-5 5",
    mic: "M12 15a3 3 0 0 0 3-3V7a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3Zm-6-3a6 6 0 0 0 12 0M12 18v3m-3 0h6",
    send: "m4 4 16 8-16 8 3-8-3-8Zm3 8h13",
    check: "m5 12 4 4L19 6",
    pause: "M8 5v14M16 5v14",
    chevron: "m7 9 5 5 5-5",
    laptop: "M4 5h16v11H4zM2 19h20",
    router: "M4 10h16v8H4zM7 7l5-3 5 3M8 14h.01M12 14h.01M16 14h.01",
    printer: "M6 9V4h12v5M6 17H4a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2h-2M6 14h12v7H6z",
    external: "M14 4h6v6m-1-5-8 8M19 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h6",
    manual: "M5 3h11a3 3 0 0 1 3 3v15H8a3 3 0 0 1-3-3V3Zm0 0v15a3 3 0 0 0 3 3M8 7h7M8 11h7",
    more: "M5 12h.01M12 12h.01M19 12h.01",
    paperclip: "m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48",
    user: "M20 21a8 8 0 0 0-16 0M12 13a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z",
  } as const;

  return <svg aria-hidden="true" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={paths[name]} /></svg>;
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
  const [selectedCategory, setSelectedCategory] = useState<DeviceCategory>("router");
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [whyOpen, setWhyOpen] = useState(false);
  const [sessionMenuOpen, setSessionMenuOpen] = useState(false);
  const [attachment, setAttachment] = useState<string | null>(null);

  const selectedDevice = deviceProfiles[selectedCategory];

  function startNewSession(category = selectedCategory) {
    setSelectedCategory(category);
    setActiveSession("new");
    setMessages([]);
    setSelectedAnswer(null);
    setState("ready");
  }

  function chooseSession(session: Session) {
    setActiveSession(session.id);
    setSelectedCategory(session.id === "printer-02" ? "printer" : session.id === "laptop-03" ? "laptop" : "router");
    setMessages(session.id === "router-01" ? initialMessages : []);
    setSelectedAnswer(null);
    setState("ready");
  }

  function submitAnswer(answer: string) {
    setSelectedAnswer(answer);
    setMessages((current) => [...current, { id: Date.now(), role: "user", text: `The WAN light is ${answer.toLowerCase()}.`, meta: "just now" }]);
    setState("thinking");
    window.setTimeout(() => setState("ready"), 650);
  }

  function submitMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = draft.trim();
    if (!text) return;
    setMessages((current) => [...current, { id: Date.now(), role: "user", text: attachment ? `${text} · ${attachment}` : text, meta: "just now" }]);
    setDraft("");
    setAttachment(null);
    setState("thinking");
    window.setTimeout(() => setState("ready"), 650);
  }

  function handleAttachment(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) setAttachment(file.name);
  }

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
            </div>

            <div className="message-list" aria-live="polite">
              {messages.length === 0 && <div className="empty-thread"><h2>Describe the problem to start.</h2><p>Use your own words. Friday will ask for one observation at a time.</p></div>}
              {messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  {message.role === "user" && <div className="message-meta"><span>You</span><span>{message.meta}</span></div>}
                  {message.role === "user" && <p>{message.text}</p>}
                  {message.role === "assistant" && (
                    <div className="step-panel">
                      <div className="step-heading"><h2>Check the WAN light</h2></div>
                      <div className="procedure-content">
                        <div className="procedure-copy"><p className="instruction">Look at the light labelled <strong>Internet</strong> or <strong>WAN</strong>. Is it off, solid, or blinking?</p></div>
                        <ManualFigure />
                      </div>
                      <div className="answer-options" aria-label="WAN light state">
                        {["Off", "Solid", "Blinking", "Not sure"].map((answer) => <button className={selectedAnswer === answer ? "selected" : ""} key={answer} type="button" aria-pressed={selectedAnswer === answer} onClick={() => submitAnswer(answer)}>{answer}</button>)}
                      </div>
                      <button className="why-button" type="button" aria-expanded={whyOpen} onClick={() => setWhyOpen((open) => !open)}>Why this step?</button>
                      {whyOpen && <p className="why-copy">The WAN light tells us whether the router sees the upstream connection. Checking it first avoids changing settings unnecessarily.</p>}
                      <div className="source-line"><Icon name="manual" /><a href="#source">Archer C6 User Guide · p. 42 · LED descriptions</a><Icon name="external" /></div>
                    </div>
                  )}
                </article>
              ))}
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

        <aside className="diagnostic-rail" aria-label="Evidence ledger">
          <div className="rail-header"><h2>What we know</h2><button className="rail-toggle" type="button" aria-label="Collapse evidence ledger"><Icon name="chevron" /></button></div>
          <div className="rail-section"><div className="rail-label">OBSERVED</div><ul className="observation-list"><li><span className="observation-dot done" /> <span>Router has power</span></li><li><span className="observation-dot done" /> <span>Wi-Fi network is visible</span></li></ul></div>
          <div className="rail-section"><div className="rail-label">{selectedAnswer ? "OBSERVED" : "NEED TO VERIFY"}</div><ul className="observation-list"><li className={selectedAnswer ? "observation-known" : ""}><span className={`observation-dot ${selectedAnswer ? "done" : "pending"}`} /> <span>{selectedAnswer ? `WAN light ${selectedAnswer.toLowerCase()}` : "WAN light state"}</span></li></ul></div>
          <div className="rail-section evidence-section"><div className="rail-label">MANUAL EVIDENCE</div><div className="evidence-card"><strong>Archer C6 User Guide</strong><span>Page 42 · LED descriptions</span><ManualFigure compact /><a href="#source">Open page 42 <Icon name="arrow" /></a></div></div>
        </aside>
      </div>
    </main>
  );
}
