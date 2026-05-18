// Strip ANSI escape codes (colors, cursor movement, etc.)
function stripAnsi(str: string): string {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, "").replace(/\x1b\][^\x07]*\x07/g, "");
}

// Handle carriage returns: \r without \n causes line overwrite — keep only the last segment
function processCarriageReturns(line: string): string {
  const parts = line.split("\r");
  return parts[parts.length - 1];
}

function colorizeLine(line: string, idx: number) {
  // Status highlights
  if (/\b(RUNNING|SUCCEEDED|FAILED|ERROR|CANCELLED)\b/i.test(line)) {
    line = line.replace(/\b(RUNNING|SUCCEEDED|FAILED|ERROR|CANCELLED)\b/gi, (m) => {
      let cls = "";
      if (/RUNNING/i.test(m)) cls = "term-status-running";
      else if (/SUCCEEDED/i.test(m)) cls = "term-status-succeeded";
      else cls = "term-status-failed";
      return `<span class='${cls}'>${m}</span>`;
    });
  }
  // Progress bars: | 3/8 [02:07<02:38, 31.72s/it] 50%|
  line = line.replace(/\|([^|]*)\|/g, (m) => {
    // Try to extract percent
    const percentMatch = m.match(/(\d{1,3})%/);
    const percent = percentMatch ? Math.min(100, Math.max(0, parseInt(percentMatch[1], 10))) : null;
    return `<span class='term-bar-outer'><span class='term-bar-inner' style='width:${percent ?? 100}%;'></span>${m}</span>`;
  });
  // Percentages
  line = line.replace(/(\d{1,3})%/g, "<span class='term-percent'>$1%</span>");
  // Times
  line = line.replace(/\b(\d{1,2}:\d{2}(?::\d{2})?)\b/g, "<span class='term-time'>$1</span>");
  // Highlight numbers in brackets
  line = line.replace(/\[(\d+\/\d+)\]/g, "<span class='term-count'>[$1]</span>");
  return <div key={idx} dangerouslySetInnerHTML={{ __html: line }} />;
}

export function TerminalLog({ text }: { text: string }) {
  // Split and colorize each line, stripping ANSI codes first
  return (
    <div className="terminal-log">
      {text.split(/\n/).map((rawLine, idx) => {
        const line = processCarriageReturns(stripAnsi(rawLine));
        return line.trim() ? colorizeLine(line, idx) : <div key={idx} className="term-blank-line" />;
      })}
    </div>
  );
}
