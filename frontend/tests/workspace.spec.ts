import { expect, test } from "@playwright/test";

const devices = [
  { manufacturer: "TP-Link", model: "Archer C6", category: "router" },
  { manufacturer: "NETGEAR", model: "R7000", category: "router" },
  { manufacturer: "HP", model: "EliteBook 855 G8", category: "laptop" },
  { manufacturer: "HP", model: "LaserJet Pro M404", category: "printer" },
];

test.beforeEach(async ({ page }) => {
  await page.route("**/v1/devices", route => route.fulfill({ json: { devices } }));
  // Deterministic UI fixture: emits visible text before the final structured
  // event. No paid provider, microphone service, or real session is contacted.
  await page.addInitScript(() => {
    const original = window.fetch;
    window.fetch = async (input, init) => {
      if (!String(input).endsWith("/v1/troubleshoot/stream")) return original(input, init);
      const request = JSON.parse(String(init?.body));
      const answer = "Let’s check the Internet light. Is it off, solid, or blinking?";
      const encoder = new TextEncoder();
      return new Response(new ReadableStream({ start(controller) {
        const send = (event: unknown) => controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
        send({ type: "token", text: answer });
        setTimeout(() => {
          send({ type: "complete", response: {
            session_id: request.session_id, status: "ready", answer,
            awaiting_observation: true, images: [], evidence: [], observations: [], missing_observations: [], facts: {}, fact_history: {},
            citations: [{ chunk_id: "fixture", document_id: "fixture", document_title: "Archer C6 User Guide", manufacturer: "TP-Link", model: "Archer C6", document_version: "test", page: 15, section: "Illustrative test evidence", source_url: "https://example.test/manual" }],
            step: { step_id: "light", title: "Internet light", instruction: answer, question: "What does it show?", source_ids: ["fixture"], options: [{ id: "off", label: "Off" }, { id: "solid", label: "Solid" }] },
            retrieval: { abstained: false, timings_ms: {} },
          }});
          controller.close();
        }, 900);
      }}), { headers: { "Content-Type": "text/event-stream" } });
    };
  });
});

test("landing demonstration and device controls work", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Look inside" }).click();
  await expect(page.getByRole("button", { name: "Assemble view" })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("tab", { name: "Printer", exact: true }).click();
  await page.getByRole("button", { name: "Ready", exact: true }).click();
  await expect(page.getByText("Observation selected: Ready.")).toBeVisible();
  await page.getByRole("button", { name: "Preview voice interaction" }).click();
  await expect(page.getByRole("button", { name: "Pause animation" })).toBeVisible();
  await page.getByRole("link", { name: "Open Friday", exact: true }).first().click();
  await expect(page).toHaveURL(/\/app$/);
});

test("stream is visible before completion and sessions retain responses", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/app");
  await page.getByRole("button", { name: "Wi-Fi connects, but there’s no internet" }).click();
  await expect(page.getByRole("textbox")).toHaveValue("Wi-Fi connects, but there’s no internet");
  await page.getByRole("button", { name: "Send observation" }).click();
  await expect(page.locator(".streaming-response")).toContainText("Let’s check the Internet light.");
  await expect(page.getByRole("button", { name: "Off", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Off", exact: true }).click();
  await expect(page.locator(".assistant-message")).toHaveCount(2);
  await expect(page.locator(".answer-options").first().getByRole("button", { name: "Off" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("button", { name: "Copy response" })).toHaveCount(2);
  await page.getByRole("button", { name: "Copy response" }).first().click();
  await expect(page.getByRole("button", { name: "Response copied" })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain("Internet light");
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await expect(page.getByRole("button", { name: "Try again", exact: true })).toBeEnabled();
  await page.reload();
  await page.locator(".session-item").first().click();
  await expect(page.locator(".assistant-message")).toHaveCount(3);
  await page.getByRole("button", { name: "New session", exact: true }).click();
  await expect(page.getByRole("heading", { name: "What isn’t working?" })).toBeVisible();
  await page.locator(".session-item").first().click();
  await expect(page.locator(".assistant-message")).toHaveCount(3);
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight)).toBe(true);
});

test("mobile navigation exposes models and evidence with keyboard dismissal", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/app");
  await page.getByRole("button", { name: "Open sessions and devices" }).click();
  await page.getByRole("combobox", { name: "Model", exact: true }).selectOption("R7000");
  await expect(page.locator(".case-context")).toContainText("NETGEAR R7000");
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Evidence & context" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("heading", { name: "Evidence & context" })).not.toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("failed requests display a recovery action", async ({ page }) => {
  await page.addInitScript(() => { const previous = window.fetch; window.fetch = async (input, init) => String(input).endsWith("/v1/troubleshoot/stream") ? new Response(null, { status: 503 }) : previous(input, init); });
  await page.goto("/app");
  await page.getByRole("textbox").fill("My router has no internet.");
  await page.getByRole("button", { name: "Send observation" }).click();
  await expect(page.locator(".api-error")).toContainText("Couldn't check the manuals");
  await expect(page.locator(".api-error").getByRole("button", { name: "Try again" })).toBeVisible();
  await page.screenshot({ path: "../.impeccable/review/request-error.png" });
});

test("voice controls pause, resume and end without leaving the workspace", async ({ page, context }) => {
  await context.grantPermissions(["microphone"]);
  await page.routeWebSocket("**/v1/voice", socket => {
    socket.onMessage(message => {
      const event = JSON.parse(String(message));
      if (event.type === "session.start") socket.send(JSON.stringify({ type: "session.ready", session_id: event.session_id }));
    });
  });
  await page.goto("/app");
  await page.getByRole("button", { name: "Talk to Friday", exact: true }).click();
  await page.getByRole("button", { name: "Pause microphone", exact: true }).click();
  await expect(page.getByText("Microphone paused", { exact: true })).toBeVisible();
  await page.screenshot({ path: "../.impeccable/review/voice-paused.png" });
  await page.getByRole("button", { name: "Resume microphone", exact: true }).click();
  await expect(page.getByRole("button", { name: "Pause microphone", exact: true })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("button", { name: "Open sessions and devices" })).toBeVisible();
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.waitForTimeout(300); // Let the viewport resize and drawer transition settle before capture.
  await page.screenshot({ path: "../.impeccable/review/mobile-voice.png" });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true);
  await page.getByRole("button", { name: "End voice", exact: true }).click();
  await expect(page.getByRole("textbox")).toBeVisible();
});
