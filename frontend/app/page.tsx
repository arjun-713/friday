import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="app-shell">
      <header className="topbar">
        <Link className="wordmark" href="/" aria-label="Friday home">
          <span className="wordmark-mark" aria-hidden="true">
            F
          </span>
          <span>friday</span>
        </Link>
        <div className="topbar-center">
          <span className="topbar-kicker">EVIDENCE-GROUNDED TROUBLESHOOTING</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", paddingRight: 22 }}>
          <Link
            href="/app"
            style={{
              fontSize: 12,
              fontWeight: 700,
              textDecoration: "none",
              border: "1px solid var(--border-strong)",
              borderRadius: 6,
              padding: "8px 12px",
              background: "var(--text)",
              color: "#fff",
            }}
          >
            Open casebook
          </Link>
        </div>
      </header>

      <div
        style={{
          maxWidth: 960,
          margin: "0 auto",
          padding: "clamp(28px, 5vw, 72px) clamp(18px, 5vw, 40px)",
          width: "100%",
        }}
      >
        <p className="case-context">LAPTOPS · WI-FI ROUTERS · PRINTERS</p>
        <h1
          style={{
            fontSize: "clamp(32px, 4vw, 52px)",
            letterSpacing: "-0.05em",
            lineHeight: 1.05,
            margin: "10px 0 0",
          }}
        >
          Describe what stopped working. Get one safe check backed by the manual.
        </h1>
        <p style={{ maxWidth: "62ch", color: "var(--muted)", fontSize: 15, lineHeight: 1.6 }}>
          Friday searches official manufacturer manuals, preserves your diagnostic context, and
          returns the safest useful next step with a document, page, and section citation. If the
          manuals cannot support a safe answer, it abstains instead of inventing a repair.
        </p>

        <div style={{ display: "flex", gap: 10, marginTop: 22, flexWrap: "wrap" }}>
          <Link
            href="/app"
            style={{
              fontSize: 13,
              fontWeight: 700,
              textDecoration: "none",
              borderRadius: 6,
              padding: "11px 16px",
              background: "var(--accent)",
              color: "#fff",
            }}
          >
            Start troubleshooting →
          </Link>
          <a
            href="http://localhost:8000/health"
            style={{
              fontSize: 13,
              fontWeight: 700,
              textDecoration: "none",
              borderRadius: 6,
              padding: "11px 16px",
              border: "1px solid var(--border-strong)",
              color: "var(--text)",
              background: "var(--surface-raised)",
            }}
          >
            Check API health
          </a>
        </div>

        <ol className="workflow-preview" style={{ marginTop: 36 }}>
          <li>
            <span>01</span>Select your device from manuals we actually have
          </li>
          <li>
            <span>02</span>Describe the symptom in your own words
          </li>
          <li>
            <span>03</span>Complete one cited diagnostic check at a time
          </li>
          <li>
            <span>04</span>Report the result and narrow the cause
          </li>
        </ol>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: 12,
            marginTop: 28,
          }}
        >
          <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 16, background: "var(--surface-raised)" }}>
            <strong style={{ fontSize: 13 }}>Laptops / Desktops</strong>
            <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 12, lineHeight: 1.5 }}>
              Dell, HP, Lenovo service and owner manuals with page-level citations.
            </p>
          </div>
          <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 16, background: "var(--surface-raised)" }}>
            <strong style={{ fontSize: 13 }}>Wi-Fi Routers</strong>
            <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 12, lineHeight: 1.5 }}>
              ASUS, Linksys, NETGEAR, TP-Link setup and connectivity procedures.
            </p>
          </div>
          <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 16, background: "var(--surface-raised)" }}>
            <strong style={{ fontSize: 13 }}>Printers</strong>
            <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 12, lineHeight: 1.5 }}>
              Brother, Canon, Epson, HP error codes, paper paths, and print quality.
            </p>
          </div>
        </div>

        <p style={{ marginTop: 24, color: "var(--muted)", fontSize: 12, lineHeight: 1.6, maxWidth: "70ch" }}>
          Every technical instruction cites its manual source. Voice and text share the same
          troubleshooting pipeline. Raw audio is not stored by default. Medical, automotive,
          aviation, high-voltage, and autonomous-repair questions are out of scope.
        </p>
      </div>
    </main>
  );
}
