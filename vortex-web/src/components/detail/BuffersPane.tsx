// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useState } from 'react';
import type { LayoutTreeNode } from '../swimlane/types';
import { formatBytes, parseArrayNodeId } from '../swimlane/utils';
import { useVortexFile } from '../../contexts/VortexFileContext';

interface BuffersPaneProps {
  node: LayoutTreeNode;
}

export function BuffersPane({ node }: BuffersPaneProps) {
  const { fetchArrayBuffer } = useVortexFile();
  const bufferLengths = node.bufferLengths ?? [];
  const [selectedBuffer, setSelectedBuffer] = useState<number | null>(null);
  const [bufferData, setBufferData] = useState<Uint8Array | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const bufferNames: string[] = bufferLengths.map((_, i) =>
    node.bufferNames?.[i] ?? `buffer ${i}`,
  );

  const loadBuffer = useCallback(
    async (index: number) => {
      setSelectedBuffer(index);
      setBufferData(null);
      setError(null);
      setLoading(true);
      try {
        const { layoutNodeId, arrayPath } = parseArrayNodeId(node.id);
        const data = await fetchArrayBuffer(layoutNodeId, arrayPath, index);
        setBufferData(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [node.id, fetchArrayBuffer],
  );

  // Auto-select first buffer when node changes.
  useEffect(() => {
    setBufferData(null);
    setError(null);
    if (bufferLengths.length > 0) {
      loadBuffer(0);
    } else {
      setSelectedBuffer(null);
    }
  }, [node.id, bufferLengths.length, loadBuffer]);

  if (bufferLengths.length === 0) {
    return (
      <div className="text-xs text-vortex-grey-dark p-2.5">
        No buffers for this node.
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Buffer list — left side */}
      <div className="w-[160px] flex-shrink-0 border-r border-vortex-grey-light/40 dark:border-white/[0.06] overflow-y-auto">
        {bufferLengths.map((len, i) => (
          <button
            key={i}
            className={`w-full text-left px-2.5 py-1.5 text-xs border-b border-vortex-grey-light/20 dark:border-white/[0.04] hover:bg-vortex-black/[0.03] dark:hover:bg-white/[0.04] ${
              selectedBuffer === i
                ? 'bg-vortex-light-blue/10 text-vortex-fg-light dark:text-vortex-fg'
                : 'text-vortex-grey-dark'
            }`}
            onClick={() => loadBuffer(i)}
          >
            <div className="font-medium text-vortex-fg-light dark:text-vortex-fg text-[11px]">
              {bufferNames[i]}
            </div>
            <div className="text-[10px] font-mono tabular-nums">{formatBytes(len)}</div>
          </button>
        ))}
      </div>

      {/* Hex viewer — right side */}
      <div className="flex-1 overflow-auto p-2">
        {selectedBuffer === null && (
          <div className="text-xs text-vortex-grey-dark">Select a buffer to view its contents.</div>
        )}
        {loading && (
          <div className="text-xs text-vortex-grey-dark">Loading…</div>
        )}
        {error && (
          <div className="text-xs text-red-500">Error: {error}</div>
        )}
        {bufferData && <HexView data={bufferData} />}
      </div>
    </div>
  );
}

function HexView({ data }: { data: Uint8Array }) {
  const bytesPerRow = 16;
  const rows: { offset: number; hex: string[]; ascii: string }[] = [];

  for (let i = 0; i < data.length; i += bytesPerRow) {
    const slice = data.subarray(i, Math.min(i + bytesPerRow, data.length));
    const hex: string[] = [];
    let ascii = '';
    for (let j = 0; j < bytesPerRow; j++) {
      if (j < slice.length) {
        hex.push(slice[j].toString(16).padStart(2, '0'));
        ascii += slice[j] >= 0x20 && slice[j] < 0x7f ? String.fromCharCode(slice[j]) : '.';
      } else {
        hex.push('  ');
        ascii += ' ';
      }
    }
    rows.push({ offset: i, hex, ascii });
  }

  // Limit display to first 1024 rows to avoid freezing
  const maxRows = 1024;
  const truncated = rows.length > maxRows;
  const visibleRows = truncated ? rows.slice(0, maxRows) : rows;

  return (
    <div className="font-mono text-[10px] leading-[16px]">
      {visibleRows.map((row) => (
        <div key={row.offset} className="flex gap-2 whitespace-pre">
          <span className="text-vortex-grey-dark w-[60px] text-right flex-shrink-0">
            {row.offset.toString(16).padStart(8, '0')}
          </span>
          <span className="text-vortex-fg-light dark:text-vortex-fg flex-shrink-0">
            {row.hex.slice(0, 8).join(' ')}
            {'  '}
            {row.hex.slice(8).join(' ')}
          </span>
          <span className="text-vortex-light-blue flex-shrink-0">
            {row.ascii}
          </span>
        </div>
      ))}
      {truncated && (
        <div className="text-vortex-grey-dark mt-1">
          … {rows.length - maxRows} more rows ({formatBytes(data.length)} total)
        </div>
      )}
    </div>
  );
}
