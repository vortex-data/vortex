// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useMemo, useState } from 'react';
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

  const bufferNames: string[] = bufferLengths.map((_, i) => node.bufferNames?.[i] ?? `buffer ${i}`);

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
    return <div className="text-xs text-vortex-grey-dark p-2.5">No buffers for this node.</div>;
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
        {loading && <div className="text-xs text-vortex-grey-dark">Loading…</div>}
        {error && <div className="text-xs text-red-500">Error: {error}</div>}
        {bufferData && <HexView data={bufferData} />}
      </div>
    </div>
  );
}

const BYTES_PER_ROW = 16;
const MAX_ROWS = 1024;

interface HexRow {
  offset: number;
  bytes: { value: number; absIndex: number }[];
}

function HexView({ data }: { data: Uint8Array }) {
  const [hovered, setHovered] = useState<number | null>(null);

  const { rows, truncated } = useMemo(() => {
    const result: HexRow[] = [];
    for (let i = 0; i < data.length; i += BYTES_PER_ROW) {
      const bytes: HexRow['bytes'] = [];
      const end = Math.min(i + BYTES_PER_ROW, data.length);
      for (let j = i; j < end; j++) {
        bytes.push({ value: data[j], absIndex: j });
      }
      result.push({ offset: i, bytes });
    }
    const isTruncated = result.length > MAX_ROWS;
    return { rows: isTruncated ? result.slice(0, MAX_ROWS) : result, truncated: isTruncated };
  }, [data]);

  const hlClass = 'bg-vortex-light-blue/20 rounded-sm';

  return (
    <div className="font-mono text-[10px] leading-[16px]">
      {rows.map((row) => (
        <div key={row.offset} className="flex gap-2 whitespace-pre">
          {/* Offset */}
          <span className="text-vortex-grey-dark w-[60px] text-right flex-shrink-0 select-none">
            {row.offset.toString(16).padStart(8, '0')}
          </span>

          {/* Hex bytes */}
          <span className="flex-shrink-0">
            {Array.from({ length: BYTES_PER_ROW }, (_, j) => {
              const b = row.bytes[j];
              if (!b)
                return (
                  <span key={j} className="inline-block w-[22px]">
                    {' '}
                  </span>
                );
              const isHl = hovered === b.absIndex;
              const gap = j === 8 ? ' ' : '';
              return (
                <span key={j}>
                  {gap}
                  <span
                    className={`inline-block w-[22px] text-center text-vortex-fg-light dark:text-vortex-fg cursor-default ${isHl ? hlClass : ''}`}
                    onMouseEnter={() => setHovered(b.absIndex)}
                    onMouseLeave={() => setHovered(null)}
                  >
                    {b.value.toString(16).padStart(2, '0')}
                  </span>
                </span>
              );
            })}
          </span>

          {/* ASCII */}
          <span className="flex-shrink-0">
            {Array.from({ length: BYTES_PER_ROW }, (_, j) => {
              const b = row.bytes[j];
              if (!b) return <span key={j}> </span>;
              const isHl = hovered === b.absIndex;
              const ch = b.value >= 0x20 && b.value < 0x7f ? String.fromCharCode(b.value) : '.';
              const color = ch === '.' ? 'text-vortex-grey-dark' : 'text-vortex-light-blue';
              return (
                <span
                  key={j}
                  className={`cursor-default ${color} ${isHl ? hlClass : ''}`}
                  onMouseEnter={() => setHovered(b.absIndex)}
                  onMouseLeave={() => setHovered(null)}
                >
                  {ch}
                </span>
              );
            })}
          </span>
        </div>
      ))}
      {truncated && (
        <div className="text-vortex-grey-dark mt-1">
          … {Math.ceil(data.length / BYTES_PER_ROW) - MAX_ROWS} more rows (
          {formatBytes(data.length)} total)
        </div>
      )}
    </div>
  );
}
