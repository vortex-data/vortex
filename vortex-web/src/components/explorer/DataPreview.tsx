// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useEffect, useRef, useState } from 'react';
import { tableFromIPC } from 'apache-arrow';
import { useSelection } from '../../contexts/SelectionContext';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { getNodeDisplayName } from '../swimlane/utils';

const ROW_LIMIT = 200;

interface PreviewTable {
  columns: string[];
  rows: string[][];
}

function decodePreview(ipcBytes: Uint8Array): PreviewTable {
  const table = tableFromIPC(ipcBytes);
  const columns = table.schema.fields.map((f) => f.name);
  const rows: string[][] = [];
  for (let i = 0; i < table.numRows; i++) {
    const row: string[] = [];
    for (const col of columns) {
      const val = table.getChild(col)?.get(i);
      row.push(val == null ? 'null' : String(val));
    }
    rows.push(row);
  }
  return { columns, rows };
}

export function DataPreview() {
  const { state: selection } = useSelection();
  const { previewData } = useVortexFile();
  const [preview, setPreview] = useState<PreviewTable | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const nodeId = selection.selectedNodeId;

  useEffect(() => {
    if (!nodeId) {
      setPreview(null);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    previewData(nodeId, ROW_LIMIT)
      .then((ipcBytes) => {
        if (!cancelled) {
          setPreview(decodePreview(ipcBytes));
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [nodeId, previewData]);

  return (
    <div ref={containerRef} className="h-full flex flex-col bg-vortex-white dark:bg-vortex-black">
      <div className="flex items-center px-3 py-1 border-b border-vortex-grey-light/40 dark:border-white/[0.06] flex-shrink-0">
        <span className="text-[10px] font-medium text-vortex-grey-dark uppercase tracking-wider">
          Preview
        </span>
        {selection.selectedNode && (
          <span className="ml-2 text-[10px] text-vortex-light-blue">
            {getNodeDisplayName(selection.selectedNode)}
          </span>
        )}
        {preview && (
          <span className="ml-auto text-[10px] text-vortex-grey-dark">
            {preview.rows.length}{preview.rows.length >= ROW_LIMIT ? '+' : ''} rows
          </span>
        )}
      </div>

      <div className="flex-1 overflow-auto">
        {!nodeId && (
          <div className="flex items-center justify-center h-full text-[11px] text-vortex-grey-dark">
            Select a node to preview data
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center h-full text-[11px] text-vortex-grey-dark">
            Loading preview…
          </div>
        )}

        {error && (
          <div className="flex items-center justify-center h-full text-[11px] text-red-500 px-4 text-center">
            {error}
          </div>
        )}

        {!loading && !error && preview && (
          <table className="w-full text-[10px] font-mono border-collapse">
            <thead className="sticky top-0 z-10">
              <tr className="bg-vortex-grey-lightest dark:bg-vortex-black">
                <th className="px-2 py-0.5 text-right text-vortex-grey-dark font-normal border-r border-vortex-grey-light/30 dark:border-white/[0.06] w-8">
                  #
                </th>
                {preview.columns.map((col) => (
                  <th
                    key={col}
                    className="px-2 py-0.5 text-left font-medium text-vortex-fg-light dark:text-vortex-fg border-r border-vortex-grey-light/30 dark:border-white/[0.06] last:border-r-0"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, ri) => (
                <tr
                  key={ri}
                  className="border-b border-vortex-grey-light/20 dark:border-white/[0.03] hover:bg-vortex-grey-lightest/50 dark:hover:bg-white/[0.02]"
                >
                  <td className="px-2 py-0.5 text-right text-vortex-grey-dark border-r border-vortex-grey-light/30 dark:border-white/[0.06] tabular-nums">
                    {ri}
                  </td>
                  {row.map((cell, ci) => (
                    <td
                      key={ci}
                      className="px-2 py-0.5 text-vortex-fg-light dark:text-vortex-fg/80 truncate max-w-[200px] border-r border-vortex-grey-light/30 dark:border-white/[0.06] last:border-r-0"
                      title={cell}
                    >
                      {cell === 'null' ? (
                        <span className="text-vortex-grey-dark italic">null</span>
                      ) : (
                        cell
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
