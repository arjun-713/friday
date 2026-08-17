class FridayAudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.samples = [];
    this.chunkSamples = Math.round(this.targetRate / 10);
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    const ratio = sampleRate / this.targetRate;
    for (let index = 0; index < input.length; index += ratio) {
      const start = Math.floor(index);
      const end = Math.min(Math.floor(index + ratio), input.length);
      let total = 0;
      for (let cursor = start; cursor < end; cursor += 1) total += input[cursor];
      this.samples.push(total / Math.max(1, end - start));
    }
    while (this.samples.length >= this.chunkSamples) {
      const chunk = this.samples.splice(0, this.chunkSamples);
      const pcm = new Int16Array(chunk.length);
      for (let index = 0; index < chunk.length; index += 1) {
        const value = Math.max(-1, Math.min(1, chunk[index]));
        pcm[index] = value < 0 ? value * 0x8000 : value * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("friday-audio-capture", FridayAudioCaptureProcessor);
