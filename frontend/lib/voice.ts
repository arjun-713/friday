import { API_BASE_URL, type TroubleshootingResponse } from "./api";

export type VoiceEvent =
  | { type: "voice.connecting" }
  | { type: "voice.ready"; sample_rate: number }
  | { type: "session.ready"; session_id: string }
  | { type: "speech.start" }
  | { type: "speech.end" }
  | { type: "transcript.partial"; text: string }
  | { type: "transcript.final"; text: string }
  | { type: "assistant.token"; text: string }
  | { type: "assistant.complete"; response: TroubleshootingResponse }
  | { type: "assistant.audio"; audio: string; sample_rate: number }
  | { type: "assistant.audio_complete" }
  | { type: "assistant.cancelled" }
  | { type: "retrieval"; retrieval: Record<string, unknown> }
  | { type: "voice.error"; message: string }
  | { type: "voice.closed"; message: string };

export type VoiceSession = {
  sessionId: string;
  manufacturer?: string;
  model?: string;
};

function voiceUrl(): string {
  const url = new URL(API_BASE_URL);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = "/v1/voice";
  return url.toString();
}

function encodePcm(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) binary += String.fromCharCode(bytes[index]);
  return window.btoa(binary);
}

function decodePcm(encoded: string): Int16Array {
  const binary = window.atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Int16Array(bytes.buffer);
}

export class FridayVoiceClient {
  private socket: WebSocket | null = null;
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: AudioWorkletNode | null = null;
  private silence: GainNode | null = null;
  private playbackSources = new Set<AudioBufferSourceNode>();
  private nextPlaybackTime = 0;
  private stopping = false;

  constructor(private readonly onEvent: (event: VoiceEvent) => void) {}

  async start(session: VoiceSession): Promise<void> {
    if (this.socket) return;
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
      throw new Error("This browser does not support microphone capture.");
    }
    const socket = new WebSocket(voiceUrl());
    this.socket = socket;
    this.stopping = false;
    try {
      await new Promise<void>((resolve, reject) => {
        socket.addEventListener("open", () => resolve(), { once: true });
        socket.addEventListener("error", () => reject(new Error("Could not connect to Friday voice.")), { once: true });
      });
    } catch (error) {
      this.stopping = true;
      socket.close();
      if (this.socket === socket) this.socket = null;
      throw error;
    }
    socket.addEventListener("message", (message) => this.handleMessage(message));
    socket.addEventListener("close", () => {
      if (this.socket === socket) this.socket = null;
      if (!this.stopping) this.onEvent({ type: "voice.closed", message: "Voice connection closed unexpectedly." });
    });
    socket.send(JSON.stringify({ type: "session.start", session_id: session.sessionId, manufacturer: session.manufacturer, model: session.model }));

    try {
      const context = new AudioContext();
      this.context = context;
      await context.audioWorklet.addModule("/audio-capture-processor.js");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      this.stream = stream;
      this.source = context.createMediaStreamSource(stream);
      this.processor = new AudioWorkletNode(context, "friday-audio-capture");
      this.silence = context.createGain();
      this.silence.gain.value = 0;
      this.processor.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        if (this.socket?.readyState === WebSocket.OPEN) {
          this.socket.send(JSON.stringify({ type: "audio", audio: encodePcm(event.data) }));
        }
      };
      this.source.connect(this.processor);
      this.processor.connect(this.silence).connect(context.destination);
      await context.resume();
    } catch (error) {
      await this.stop();
      throw error;
    }
  }

  cancelAssistant(): void {
    this.clearPlayback();
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: "assistant.cancel" }));
  }

  async stop(): Promise<void> {
    this.stopping = true;
    this.cancelAssistant();
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: "session.stop" }));
    this.socket?.close();
    this.socket = null;
    this.processor?.disconnect();
    this.source?.disconnect();
    this.silence?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.processor = null;
    this.source = null;
    this.silence = null;
    this.clearPlayback();
    await this.context?.close();
    this.context = null;
  }

  private handleMessage(message: MessageEvent<string>): void {
    try {
      const event = JSON.parse(message.data) as VoiceEvent;
      if (event.type === "assistant.audio") this.playPcm(event.audio, event.sample_rate);
      if (event.type === "assistant.cancelled") this.clearPlayback();
      this.onEvent(event);
    } catch {
      this.onEvent({ type: "voice.error", message: "Friday received an invalid voice event." });
    }
  }

  private playPcm(encoded: string, sampleRate: number): void {
    const context = this.context;
    if (!context) return;
    const pcm = decodePcm(encoded);
    const audio = context.createBuffer(1, pcm.length, sampleRate);
    const channel = audio.getChannelData(0);
    for (let index = 0; index < pcm.length; index += 1) channel[index] = pcm[index] / 0x8000;
    const source = context.createBufferSource();
    source.buffer = audio;
    source.connect(context.destination);
    source.onended = () => this.playbackSources.delete(source);
    const start = Math.max(context.currentTime + 0.02, this.nextPlaybackTime);
    source.start(start);
    this.nextPlaybackTime = start + audio.duration;
    this.playbackSources.add(source);
  }

  private clearPlayback(): void {
    for (const source of this.playbackSources) source.stop();
    this.playbackSources.clear();
    this.nextPlaybackTime = 0;
  }
}
