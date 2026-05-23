import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

interface ActionItem {
  key: string;
  label: string;
  icon: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  disabledTitle?: string;
  danger?: boolean;
}

interface Props {
  open: boolean;
  onClose: () => void;
  items: ActionItem[];
}

export function WorkspaceActionsMenu({ open, onClose, items }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const handleDocClick = (e: MouseEvent) => {
      if (!ref.current) return;
      if (e.target instanceof Node && ref.current.contains(e.target)) return;
      onClose();
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    // Defer so the click that opened the menu doesn't immediately close it.
    const t = window.setTimeout(() => {
      document.addEventListener("mousedown", handleDocClick);
      document.addEventListener("keydown", handleKey);
    }, 0);
    return () => {
      window.clearTimeout(t);
      document.removeEventListener("mousedown", handleDocClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="ws-actions-menu" ref={ref} role="menu">
      {items.map((item) => {
        const showBubble = !!(item.disabled && item.disabledTitle);
        return (
          <div
            key={item.key}
            className={`ws-actions-menu-row${showBubble ? " has-bubble" : ""}`}
            data-bubble={showBubble ? item.disabledTitle : undefined}
          >
            <button
              type="button"
              role="menuitem"
              className={`ws-actions-menu-item${item.danger ? " danger" : ""}`}
              disabled={item.disabled}
              title={item.disabled ? undefined : item.label}
              onClick={() => {
                if (item.disabled) return;
                item.onClick();
                onClose();
              }}
            >
              <span className="ws-actions-menu-icon">{item.icon}</span>
              <span className="ws-actions-menu-label">{item.label}</span>
            </button>
          </div>
        );
      })}
    </div>
  );
}
