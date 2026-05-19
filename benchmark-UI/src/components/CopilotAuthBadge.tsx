import { useEffect, useState } from "react";

interface AuthStatus {
  ok: boolean;
  detail: string;
}

interface Props {
  bridgeUrl: string;
}

export function CopilotAuthBadge({ bridgeUrl }: Props) {
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [checking, setChecking] = useState(false);

  const check = async () => {
    setChecking(true);
    try {
      const res = await fetch(`${bridgeUrl}/api/copilot/auth/status`);
      if (!res.ok) {
        setStatus({ ok: false, detail: `Bridge returned ${res.status}` });
        return;
      }
      setStatus((await res.json()) as AuthStatus);
    } catch (e) {
      setStatus({
        ok: false,
        detail: `Cannot reach bridge at ${bridgeUrl}. Run 'npm run start' in copilot-bridge.`,
      });
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    void check();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bridgeUrl]);

  const dotColor = status?.ok ? "#19a974" : "#d4453e";

  return (
    <div
      className="copilot-auth-badge"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 10px",
        borderRadius: 6,
        background: "var(--panel-bg, rgba(127,127,127,0.08))",
        fontSize: 12,
      }}
    >
      <span
        style={{
          display: "inline-block",
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: dotColor,
        }}
      />
      <span>
        {status === null
          ? "Checking Copilot..."
          : status.ok
            ? "Copilot signed in"
            : "Copilot not authenticated — run 'copilot login'"}
      </span>
      <button
        type="button"
        className="btn"
        style={{ padding: "2px 8px", fontSize: 11 }}
        onClick={check}
        disabled={checking}
      >
        {checking ? "..." : "Recheck"}
      </button>
      {status && !status.ok && status.detail && (
        <small style={{ opacity: 0.65 }}>{status.detail}</small>
      )}
    </div>
  );
}
