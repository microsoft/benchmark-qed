// Tracks pending interactive requests from the agent (ask_user, elicitation,
// permission) and exposes a promise that resolves when the browser posts an
// answer back. Each request has a unique requestId so multiple can be in flight.

import { randomUUID } from "node:crypto";

export class PendingRequests {
  constructor() {
    /** @type {Map<string, { kind: string, sessionId: string, payload: any, resolve: (v:any)=>void, reject: (e:any)=>void }>} */
    this.byId = new Map();
  }

  /**
   * Register a pending request. Returns { requestId, promise }.
   * The caller (handler bridge) awaits `promise` and returns the resolved value
   * to the SDK.
   */
  create(sessionId, kind, payload) {
    const requestId = randomUUID();
    const promise = new Promise((resolve, reject) => {
      this.byId.set(requestId, { kind, sessionId, payload, resolve, reject });
    });
    return { requestId, promise };
  }

  resolve(requestId, value) {
    const entry = this.byId.get(requestId);
    if (!entry) return false;
    this.byId.delete(requestId);
    entry.resolve(value);
    return true;
  }

  reject(requestId, error) {
    const entry = this.byId.get(requestId);
    if (!entry) return false;
    this.byId.delete(requestId);
    entry.reject(error);
    return true;
  }

  /** Reject all pending requests for a session (called on session disconnect). */
  rejectSession(sessionId, error) {
    for (const [id, entry] of this.byId) {
      if (entry.sessionId === sessionId) {
        this.byId.delete(id);
        entry.reject(error);
      }
    }
  }

  list(sessionId) {
    const out = [];
    for (const [id, entry] of this.byId) {
      if (entry.sessionId === sessionId) {
        out.push({ requestId: id, kind: entry.kind, payload: entry.payload });
      }
    }
    return out;
  }
}
