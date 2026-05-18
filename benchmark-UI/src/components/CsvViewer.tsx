import { useMemo } from "react";
import Papa from "papaparse";

interface Props {
  content: string;
  delimiter?: string;
}

export function CsvViewer({ content, delimiter }: Props) {
  const { headers, rows, error } = useMemo(() => {
    const result = Papa.parse<string[]>(content, {
      delimiter: delimiter ?? "",
      skipEmptyLines: true,
    });
    if (result.errors.length > 0 && result.data.length === 0) {
      return { headers: [], rows: [], error: result.errors[0].message };
    }
    const data = result.data;
    const headers = data.length > 0 ? data[0] : [];
    const rows = data.slice(1);
    return { headers, rows, error: null };
  }, [content, delimiter]);

  if (error) return <div className="error">CSV parse error: {error}</div>;

  return (
    <div className="csv-viewer">
      <div className="csv-info">
        {rows.length} rows × {headers.length} columns
      </div>
      <div className="csv-scroll">
        <table className="csv-table">
          <thead>
            <tr>
              {headers.map((h, i) => (
                <th key={i}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => (
              <tr key={ri}>
                {row.map((cell, ci) => (
                  <td key={ci}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
