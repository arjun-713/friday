"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowRightIcon, ArrowUpRightIcon, BookOpenIcon, CheckIcon, ChevronDownIcon, ComputerDesktopIcon, MicrophoneIcon, MinusIcon, PlusIcon, PrinterIcon, WifiIcon, XMarkIcon, Bars3Icon, PauseIcon, PlayIcon } from "@heroicons/react/24/outline";
import Brand, { BrandMark } from "./Brand";
import "../app/landing.css";

const DeviceScene = dynamic(() => import("./DeviceScene"), { ssr: false, loading: () => <div className="scene-loading"><WifiIcon/><span>Loading device view</span></div> });

const examples = [
  { label: "Wi-Fi router", icon: WifiIcon, user: "My phone connects to Wi-Fi, but nothing loads.", answer: "Let’s check the connection between your router and modem. What does the Internet light on the router look like?", device: "TP-Link Archer C6", source: "Archer C6 User Guide", section: "Internet connection troubleshooting", choices: ["Off", "Solid", "Blinking"] },
  { label: "Laptop", icon: ComputerDesktopIcon, user: "My laptop is plugged in, but the battery isn’t charging.", answer: "Let’s see whether the laptop detects its power adapter. What does the battery icon say when you connect the charger?", device: "Laptop / Desktop", source: "Manufacturer service manual", section: "Power and battery troubleshooting", choices: ["Charging", "Plugged in", "No change"] },
  { label: "Printer", icon: PrinterIcon, user: "The printer is on, but my document won’t print.", answer: "Let’s start with the printer itself. Is there a message on its display, or does it say Ready?", device: "Printer", source: "Manufacturer user guide", section: "Printing troubleshooting", choices: ["Ready", "An error message", "No display"] },
];

function SignalTrace() {
  return <svg className="signal-trace" viewBox="0 0 560 460" fill="none" aria-hidden="true"><path d="M60 90H160Q180 90 180 110V180Q180 200 200 200H290Q310 200 310 220V310Q310 330 330 330H495" stroke="currentColor" strokeWidth="1"/><path className="trace-travel" pathLength="100" d="M60 90H160Q180 90 180 110V180Q180 200 200 200H290Q310 200 310 220V310Q310 330 330 330H495" stroke="#b5e4bc" strokeWidth="2"/><circle cx="60" cy="90" r="4" fill="#b5e4bc"/><circle cx="495" cy="330" r="4" fill="#b5e4bc"/></svg>;
}

export default function Landing() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [example, setExample] = useState(0);
  const [demoAnswer, setDemoAnswer] = useState<string | null>(null);
  const [animateVoice, setAnimateVoice] = useState(false);
  const sample = examples[example];
  return (
    <main className="landing">
      <a className="skip-link" href="#landing-content">Skip to content</a>
      <header className="landing-nav">
        <Brand/>
        <nav className={menuOpen ? "landing-links is-open" : "landing-links"} aria-label="Main navigation">
          <a href="#how-it-works" onClick={() => setMenuOpen(false)}>How it works</a><a href="#devices" onClick={() => setMenuOpen(false)}>Your devices</a><a href="#voice" onClick={() => setMenuOpen(false)}>Voice & text</a>
        </nav>
        <Link className="nav-cta" href="/app">Open Friday <ArrowUpRightIcon/></Link>
        <button className="landing-menu" onClick={() => setMenuOpen(!menuOpen)} aria-label={menuOpen ? "Close navigation" : "Open navigation"} aria-expanded={menuOpen}>{menuOpen ? <XMarkIcon/> : <Bars3Icon/>}</button>
      </header>

      <section className="landing-hero" id="landing-content">
        <div className="hero-copy">
          <h1>Back to<br/>working.<br/><span>Back to life.</span></h1>
          <p>Talk through the problem.<br/>Keep your hands on the device.</p>
          <p className="hero-description">Tell Friday what stopped working. It checks your device’s manual and talks you through what to try next. Ask a question, interrupt, or switch to text.</p>
          <Link className="button-primary" href="/app"><MicrophoneIcon/>Talk to Friday <ArrowRightIcon/></Link>
          <a className="hero-watch" href="#voice"><PlayIcon/> See voice in action</a>
        </div>
        <div className="hero-stage">
          <div className="stage-coordinate"><span>Device → Evidence → Next step</span><span>Interactive illustration</span></div>
          <div className="stage-light"/>
          <SignalTrace/>
          <DeviceScene expanded={expanded}/>
          <div className="stage-question"><span className="small-avatar">You</span><p>“The Wi-Fi is on.<br/>The internet isn’t.”</p></div>
          <div className="stage-evidence"><span className="evidence-icon"><BookOpenIcon/></span><div><strong>Start with the source.</strong><span>Manual evidence → a useful next check</span></div><CheckIcon/></div>
          <div className="stage-controls"><span><i/> Explore the hardware</span><button type="button" onClick={() => setExpanded(!expanded)} aria-pressed={expanded}>{expanded ? <MinusIcon/> : <PlusIcon/>}{expanded ? "Assemble view" : "Look inside"}</button></div>
        </div>
        <div className="hero-bottom"><span>A little clarity for the devices you depend on.</span><div><span><ComputerDesktopIcon/> Computers</span><span><WifiIcon/> Routers</span><span><PrinterIcon/> Printers</span></div><a href="#how-it-works" aria-label="Scroll to how Friday works"><ChevronDownIcon/></a></div>
      </section>

      <section className="workflow-section light-section" id="how-it-works">
        <div className="section-intro"><h2>Less searching.<br/><span>More figuring it out.</span></h2><p>You don’t need to know the right technical terms. Start with what you see. Friday connects your observations to the relevant manual.</p></div>
        <div className="workflow-layout">
          <div className="workflow-steps">
            <div><span className="step-number">1</span><h3>Tell it what’s happening.</h3><p>A light that won’t stop blinking. A print job that won’t move. Your own words are enough.</p></div>
            <div><span className="step-number">2</span><h3>Find the evidence.</h3><p>Friday retrieves the relevant guidance for your device and keeps its sources attached.</p></div>
            <div><span className="step-number">3</span><h3>Work through it together.</h3><p>Try the next check, share what changed, and continue the conversation from there.</p></div>
            <Link className="text-link" href="/app">Try it with your device <ArrowUpRightIcon/></Link>
          </div>
          <div className="product-demo">
            <div className="demo-top"><div><BrandMark/><strong>Friday</strong><span>Interactive example</span></div><span className="demo-window-dots"><i/><i/><i/></span></div>
            <div className="demo-tabs" role="tablist" aria-label="Example device">{examples.map((item, index) => <button key={item.label} id={`example-tab-${index}`} role="tab" aria-selected={example === index} aria-controls="example-panel" onClick={() => { setExample(index); setDemoAnswer(null); }}><item.icon/>{item.label}</button>)}</div>
            <div className="demo-conversation" id="example-panel" role="tabpanel" aria-labelledby={`example-tab-${example}`}>
              <span className="demo-device">{sample.device}</span>
              <p className="demo-user">{sample.user}</p>
              <div className="demo-assistant"><BrandMark/><div><p>{sample.answer}</p><div className="demo-options">{sample.choices.map(choice => <button aria-pressed={demoAnswer === choice} key={choice} onClick={() => setDemoAnswer(choice)}>{choice}{demoAnswer === choice && <CheckIcon/>}</button>)}</div><div className="demo-source"><BookOpenIcon/><span>{sample.source}<small>{sample.section} · Example source</small></span></div></div></div>
              {demoAnswer && <p className="demo-continuation"><CheckIcon/> Observation selected: {demoAnswer}. <Link href="/app">Open a real session <ArrowRightIcon/></Link></p>}
            </div>
            <Link href="/app" className="demo-composer"><span>Describe what you notice…</span><MicrophoneIcon/><span className="demo-send"><ArrowRightIcon/></span></Link>
          </div>
        </div>
      </section>

      <section className="evidence-story light-section" id="evidence">
        <div className="manual-composition" aria-label="Illustration of a manual excerpt becoming a cited response">
          <div className="manual-sheet back-sheet" aria-hidden="true"/>
          <div className="manual-sheet"><div className="sheet-header"><BookOpenIcon/><span>DEVICE USER GUIDE</span></div><h3>Understanding<br/>your connection.</h3><div className="manual-line"/><div className="manual-line short"/><svg className="manual-diagram" viewBox="0 0 280 94" fill="none" role="img" aria-label="Connection diagram from modem to router to laptop"><rect x="5" y="32" width="48" height="32" rx="5"/><path d="M15 43h20m-20 10h10M54 48h41m0 0-6-5m6 5-6 5"/><rect x="100" y="32" width="60" height="32" rx="5"/><path d="M110 32V11m39 21V11m-36 40h5m9 0h5m12 0h5M162 48h38m0 0-6-5m6 5-6 5"/><rect x="209" y="21" width="58" height="39" rx="3"/><path d="M202 66h72m-65-10h58"/></svg><div className="highlighted-excerpt">Check the connection between your modem and the router’s Internet port.</div><div className="manual-line"/><div className="manual-line"/><div className="manual-line short"/><span className="sheet-caption">Illustrative excerpt · not a repair instruction</span></div>
          <div className="source-result"><CheckIcon/><div><strong>Keep the source in sight.</strong><span>Document. Section. Page.</span></div><ArrowUpRightIcon/></div>
        </div>
        <div className="evidence-story-copy"><h2>A helpful answer.<br/><span>And where it came from.</span></h2><p>When something is broken, a confident guess isn’t enough. Friday brings the manual into the conversation, so you can see the source behind a recommendation.</p><ul><li><CheckIcon/> Manufacturer documentation, retrieved locally</li><li><CheckIcon/> Page and section references alongside the answer</li><li><CheckIcon/> Clear limits when the evidence falls short</li></ul><Link className="text-link" href="/app">See the evidence for yourself <ArrowUpRightIcon/></Link></div>
      </section>

      <section className="voice-story" id="voice"><div className="voice-story-copy"><h2>Hands on the device.<br/><span>Stay in the conversation.</span></h2><p>Talk while you troubleshoot. Pause the microphone, interrupt a reply, or return to typing. Your conversation and its sources stay together.</p><Link className="button-primary" href="/app">Open Friday <ArrowRightIcon/></Link><small>Microphone access is requested only when you start voice.</small></div><div className={`voice-demo ${animateVoice ? "playing" : ""}`}><div className="voice-demo-top"><BrandMark/><span>Voice interaction preview</span></div><div className="voice-orbit"><svg viewBox="0 0 260 260" aria-hidden="true"><circle cx="130" cy="130" r="111"/><circle cx="130" cy="130" r="86"/><circle cx="130" cy="130" r="61"/></svg><MicrophoneIcon/></div><div className="voice-demo-wave" aria-hidden="true">{Array.from({ length: 43 }, (_, i) => <i key={i} style={{ height: `${10 + Math.sin(i * 1.4) ** 2 * (42 - Math.abs(i - 21))}px`, animationDelay: `${i * 31}ms` }}/>)}</div><p>{animateVoice ? "A conversation, at your pace." : "Space to think. Room to respond."}</p><button type="button" aria-pressed={animateVoice} onClick={() => setAnimateVoice(!animateVoice)}>{animateVoice ? <PauseIcon/> : <PlayIcon/>}{animateVoice ? "Pause animation" : "Preview voice interaction"}</button><small>Visual preview · no microphone or audio</small></div></section>

      <section className="device-section light-section" id="devices"><div className="section-intro"><h2>For the everyday<br/><span>“why won’t this work?”</span></h2><p>Start with a supported device. Friday uses its available manuals to help you make sense of the problem.</p></div><div className="device-directory">{[{ icon: ComputerDesktopIcon, name: "Laptops & desktops", detail: "Power, charging, startup, and connections", number: "01" }, { icon: WifiIcon, name: "Wi-Fi routers", detail: "Internet access, indicators, and setup", number: "02" }, { icon: PrinterIcon, name: "Printers", detail: "Print queues, error messages, and paper paths", number: "03" }].map(item => <Link href="/app" key={item.name}><item.icon/><h3>{item.name}</h3><p>{item.detail}</p><ArrowUpRightIcon/></Link>)}</div><p className="device-availability">Exact model availability is shown in the app’s device selector.</p></section>

      <section className="faq-section light-section"><h2>A few things<br/>before you start.</h2><div>{[{ q: "Do I need to know the technical terms?", a: "No. Describe the problem in your own words. Friday can ask for the details it needs and help you interpret what you’re seeing." }, { q: "Where do the answers come from?", a: "Friday retrieves context from locally indexed manufacturer manuals, then uses a language model to help explain it. Source references let you check the original guidance. AI responses can still make mistakes." }, { q: "Does everything run on my device?", a: "Manual retrieval and embeddings run locally on the server hosting Friday. Language generation and speech use external API providers. It is not an entirely offline assistant." }, { q: "Can I come back to a conversation?", a: "Yes. Your sessions are saved in this browser. Use the session list to return to a case, or start a new session for another problem." }].map(item => <details key={item.q}><summary>{item.q}<PlusIcon/></summary><p>{item.a}</p></details>)}</div></section>
      <footer className="landing-footer"><div className="footer-invitation"><h2>Let’s get it<br/>working again.</h2><Link className="button-primary" href="/app">Start troubleshooting <ArrowRightIcon/></Link></div><div className="footer-bottom"><Brand/><p>Your device. Its manual. A way forward.</p><a href="#landing-content">Back to top <ArrowUpRightIcon/></a></div></footer>
    </main>
  );
}
