import { useEffect, useRef } from "react";

export interface SkillMenuItem {
  key: string;
  label: string;
  description?: string;
}

interface Props {
  open: boolean;
  items: SkillMenuItem[];
  onSelect: (key: string) => void;
  onClose: () => void;
}

/**
 * Popover menu listing available Copilot skills. Mirrors the visual language
 * of WorkspaceActionsMenu (reuses its CSS) but lays out each item as a
 * stacked label + short description so the user can tell skills apart.
 */
export function SkillPickerMenu({ open, items, onSelect, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    // Defer to avoid catching the same click that opened the menu.
    const t = setTimeout(() => {
      const handler = (e: MouseEvent) => {
        if (!ref.current) return;
        if (ref.current.contains(e.target as Node)) return;
        onClose();
      };
      window.addEventListener("mousedown", handler);
      return () => window.removeEventListener("mousedown", handler);
    }, 0);
    return () => clearTimeout(t);
  }, [open, onClose]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      ref={ref}
      className="ws-actions-menu skill-picker-menu"
      role="menu"
      aria-label="Pick a Copilot skill"
    >
      {items.map((item) => (
        <button
          key={item.key}
          type="button"
          role="menuitem"
          className="ws-actions-menu-item skill-picker-item"
          onClick={() => {
            onSelect(item.key);
            onClose();
          }}
        >
          <span className="skill-picker-label">{item.label}</span>
          {item.description && (
            <span className="skill-picker-desc">{item.description}</span>
          )}
        </button>
      ))}
    </div>
  );
}
