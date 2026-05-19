// Minimal per-session SSE bus. Each session has one logical event stream
// consumed by zero or more connected browser tabs. Events are JSON-serializable.
// We also buffer the most recent events so that a UI that reconnects mid-flight
// can replay the conversation, and so that pending requests (ask_user /
// elicitation / permission) survive a transient browser reconnect.

const MAX_BUFFER = 500;

export class SseBus {
  constructor() {
    /** @type {Map<string, { buffer: any[], subscribers: Set<(e:any)=>void> }>} */
    this.streams = new Map();
  }

  ensure(sessionId) {
    let s = this.streams.get(sessionId);
    if (!s) {
      s = { buffer: [], subscribers: new Set() };
      this.streams.set(sessionId, s);
    }
    return s;
  }

  publish(sessionId, event) {
    const s = this.ensure(sessionId);
    const stamped = { ...event, ts: Date.now() };
    s.buffer.push(stamped);
    if (s.buffer.length > MAX_BUFFER) s.buffer.shift();
    for (const sub of s.subscribers) {
      try {
        sub(stamped);
      } catch {
        /* subscriber errors must not break publish */
      }
    }
  }

  subscribe(sessionId, handler, { replay = true } = {}) {
    const s = this.ensure(sessionId);
    if (replay) {
      for (const evt of s.buffer) handler(evt);
    }
    s.subscribers.add(handler);
    return () => s.subscribers.delete(handler);
  }

  drop(sessionId) {
    this.streams.delete(sessionId);
  }
}
