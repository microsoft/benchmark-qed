export interface ActivityLogEntry {
  id: string;
  timestamp: Date;
  action: string;
  details?: string;
  type: "info" | "success" | "warning" | "error";
}

import {
  ChevronRight16Regular,
  ChevronDown16Regular,
} from "@fluentui/react-icons";

interface Props {
  entries: ActivityLogEntry[];
  onClear?: () => void;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
}

export function ActivityLogPanel({ entries, onClear, collapsed = false, onToggleCollapsed }: Props) {
  return (
    <aside className={`activity-log-panel ${collapsed ? 'collapsed' : ''}`}>
      <div className="activity-log-header">
        <div className="activity-log-header-content">
          <h3>Activity Log</h3>
          {onToggleCollapsed && (
            <button
              className="activity-log-toggle"
              onClick={onToggleCollapsed}
              title={collapsed ? "Expand log" : "Collapse log"}
            >
              {collapsed ? <ChevronRight16Regular /> : <ChevronDown16Regular />}
            </button>
          )}
        </div>
        {!collapsed && onClear && entries.length > 0 && (
          <button
            className="activity-log-clear"
            onClick={onClear}
            title="Clear activity log"
            aria-label="Clear activity log"
          >
            Clear
          </button>
        )}
      </div>
      {!collapsed && (
        <div className="activity-log-list">
          {entries.length === 0 ? (
            <div className="activity-log-empty">No activity yet</div>
          ) : (
            entries.map((entry) => (
              <div key={entry.id} className={`activity-log-entry activity-log-${entry.type}`}>
                <div className="activity-log-time">
                  {entry.timestamp.toLocaleTimeString([], { 
                    hour: "2-digit", 
                    minute: "2-digit", 
                    second: "2-digit" 
                  })}
                </div>
                <div className="activity-log-action">{entry.action}</div>
                {entry.details && (
                  <div className="activity-log-details">{entry.details}</div>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </aside>
  );
}
