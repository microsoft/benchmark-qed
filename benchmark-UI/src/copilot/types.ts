// Types mirroring the copilot-bridge SSE protocol. Kept narrow on purpose —
// only the fields the UI needs are listed. See copilot-bridge/README.md for
// the wire format.

export interface AskUserPayload {
  question: string;
  choices?: string[];
  allowFreeform: boolean;
}

export interface ElicitationPayload {
  message: string;
  requestedSchema: JsonSchema;
  mode?: "form" | "url";
  elicitationSource?: string;
}

export interface PermissionPayload {
  kind: string;
  toolName?: string;
  toolCallId?: string;
  fileName?: string;
  fullCommandText?: string;
  // The SDK includes extra fields for some `kind` values (e.g. read /
  // list permissions ship `directory` or `path`, not `fileName`). Those
  // are forwarded verbatim by the bridge, so allow arbitrary additional
  // properties here.
  [extra: string]: unknown;
}

export type PendingKind = "user_input" | "elicitation" | "permission";

export interface PendingRequest<K extends PendingKind = PendingKind> {
  kind: K;
  requestId: string;
  payload: K extends "user_input"
    ? AskUserPayload
    : K extends "elicitation"
      ? ElicitationPayload
      : PermissionPayload;
}

export interface AskUserAnswer {
  answer: string;
  wasFreeform?: boolean;
}

export interface ElicitationAnswer {
  action: "accept" | "decline" | "cancel";
  content?: Record<string, unknown>;
}

export interface PermissionAnswer {
  decision: "approve-once" | "approve-for-session" | "reject";
  feedback?: string;
}

export interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  streaming?: boolean;
}

export interface ToolEvent {
  id: string;
  toolName: string;
  status: "running" | "complete";
  args?: unknown;
  result?: unknown;
  startedAt: number;
  endedAt?: number;
}

export interface SessionStatus {
  state: "connecting" | "idle" | "running" | "error" | "closed";
  error?: string;
}

export interface AuthStatus {
  ok: boolean;
  detail: string;
}

// Minimal JSON Schema subset used by elicitation forms. We pass-through what
// the SDK gives us, but type the common shapes so the form renderer can be safe.
export interface JsonSchema {
  type?: "object" | "string" | "number" | "integer" | "boolean" | "array";
  title?: string;
  description?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  enum?: Array<string | number | boolean>;
  default?: unknown;
  minLength?: number;
  maxLength?: number;
  minimum?: number;
  maximum?: number;
  format?: string;
  items?: JsonSchema;
}
