// src/network.ts
import { EMA } from './utils.js';
import type { PoseFrame } from './types.js';

type StatusCallback = (status: string) => void;

interface BufferedFrame {
  ts: number;
  frame: number;
  data: PoseFrame;
}

export class PoseStream {
  private url: string;
  private onStatus: StatusCallback;
  private ws: WebSocket | null = null;
  private buffer: BufferedFrame[] = [];
  private maxBufferMs = 150;
  private targetLatencyMs = 120;
  private clockOffset = 0; // clientNow - serverNow
  private clockFilter = new EMA(0.05, 0);

  constructor(url: string, onStatus?: StatusCallback) {
    this.url = url;
    this.onStatus = onStatus ?? (() => {});
    this.connect();
  }

  private connect(): void {
    this.ws = new WebSocket(this.url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => this.onStatus('connected');
    this.ws.onclose = () => {
      this.onStatus('disconnected, retrying…');
      setTimeout(() => this.connect(), 1000);
    };
    this.ws.onerror = () => this.onStatus('error');

    this.ws.onmessage = (ev: MessageEvent<string | ArrayBuffer>) => {
      const msg: PoseFrame = typeof ev.data === 'string'
        ? JSON.parse(ev.data)
        : JSON.parse(new TextDecoder().decode(ev.data));

      if (msg.type !== 'poseFrame') return;

      const serverTs = msg.timestamp;       // ms epoch from server
      const clientTs = Date.now();          // ms epoch on client
      const sampleOffset = clientTs - serverTs;
      this.clockOffset = this.clockFilter.update(sampleOffset);

      this.buffer.push({ ts: serverTs, frame: msg.frame, data: msg });
      this.buffer.sort((a, b) => a.ts - b.ts);

      // Trim head to cap buffer span
      const head = this.buffer[0]?.ts;
      if (head !== undefined) {
        while (this.buffer.length && (this.buffer[this.buffer.length - 1]!.ts - head) > this.maxBufferMs) {
          this.buffer.shift();
        }
      }
    };
  }

  // Returns either two frames for interpolation, or a single frame for hold.
  pollInterpolated():
    | { kind: 'interp'; a: PoseFrame; b: PoseFrame; alpha: number }
    | { kind: 'hold'; data: PoseFrame; alpha: 0 }
    | null
  {
    if (!this.buffer.length) return null;

    const clientNow = Date.now();
    const serverNow = clientNow - this.clockOffset;
    const targetServerTime = serverNow - this.targetLatencyMs;

    let i1 = -1;
    for (let i = this.buffer.length - 1; i >= 0; --i) {
      if (this.buffer[i]!.ts <= targetServerTime) { i1 = i; break; }
    }
    if (i1 === -1) {
      return { kind: 'hold', data: this.buffer[0]!.data, alpha: 0 };
    }
    if (i1 === this.buffer.length - 1) {
      return { kind: 'hold', data: this.buffer[i1]!.data, alpha: 0 };
    }

    const a = this.buffer[i1]!;
    const b = this.buffer[i1 + 1]!;
    const span = Math.max(1, b.ts - a.ts);
    const t = (targetServerTime - a.ts) / span;

    return { kind: 'interp', a: a.data, b: b.data, alpha: t };
  }

  /**
   * Get the most recently received frame (if any). This returns the latest
   * buffered PoseFrame as-is and is intended for UI/debug readout of
   * metadata (frame number, skeleton name, confidence, etc.).
   */
  public getLatestFrame(): PoseFrame | null {
    if (!this.buffer.length) return null;
    return this.buffer[this.buffer.length - 1]!.data;
  }
}
