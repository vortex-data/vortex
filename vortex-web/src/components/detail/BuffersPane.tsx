// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { LayoutTreeNode } from '../swimlane/types';
import { formatBytes, parseArrayNodeId } from '../swimlane/utils';
import { useVortexFile } from '../../contexts/VortexFileContextCore';

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
    return <p className="vx-label x-pad">This node has no buffers.</p>;
  }

  return (
    <div className="x-buffers">
      <fieldset className="x-buffer-list">
        <legend className="vx-sr-only">Buffers</legend>
        {bufferLengths.map((len, i) => (
          <button
            type="button"
            key={i}
            aria-pressed={selectedBuffer === i}
            onClick={() => loadBuffer(i)}
          >
            <span className="x-buffer-name">{bufferNames[i]}</span>
            <span className="vx-label-sm vx-num">{formatBytes(len)}</span>
          </button>
        ))}
      </fieldset>

      <div className="x-hex">
        {selectedBuffer === null && <p className="vx-label">Select a buffer to see its bytes.</p>}
        {loading && <p className="vx-label">Reading the buffer…</p>}
        {error && (
          <div role="alert">
            <p className="vx-small x-error">The buffer did not load. Select it again.</p>
            <p className="vx-label-sm">
              Reason · <span className="vx-ref">{error}</span>
            </p>
          </div>
        )}
        {bufferData && <HexView data={bufferData} />}
      </div>
    </div>
  );
}

const BYTES_PER_ROW = 16;
const MAX_ROWS = 1024;

const hex = (n: number, width: number) => n.toString(16).padStart(width, '0');
const printable = (b: number) => (b >= 0x20 && b < 0x7f ? String.fromCharCode(b) : null);

/**
 * Offset, hex, and text columns, 16 bytes a row. Hover marks a byte in both columns; the pointer
 * is read from the row, so a byte needs no handlers of its own.
 */
function HexView({ data }: { data: Uint8Array }) {
  const [hovered, setHovered] = useState<number | null>(null);
  const rows = Math.min(Math.ceil(data.length / BYTES_PER_ROW), MAX_ROWS);
  const truncated = Math.ceil(data.length / BYTES_PER_ROW) > MAX_ROWS;

  const onPointer = (e: React.PointerEvent) => {
    const at = (e.target as HTMLElement).dataset.at;
    setHovered(at === undefined ? null : Number(at));
  };

  const lines = useMemo(() => {
    const out: { offset: number; bytes: number[] }[] = [];
    for (let r = 0; r < rows; r++) {
      const offset = r * BYTES_PER_ROW;
      out.push({ offset, bytes: Array.from(data.subarray(offset, offset + BYTES_PER_ROW)) });
    }
    return out;
  }, [data, rows]);

  return (
    // Hover only marks a byte; nothing acts on it.
    <div onPointerOver={onPointer} onPointerLeave={() => setHovered(null)}>
      {lines.map(({ offset, bytes }) => (
        <div key={offset} className="x-hex-row">
          <span className="x-hex-offset">{hex(offset, 8)}</span>
          <span className="x-hex-bytes">
            {bytes.map((b, j) => (
              <span
                key={j}
                data-at={offset + j}
                className={`x-hex-byte${j === 8 ? ' is-gap' : ''}${hovered === offset + j ? ' is-hovered' : ''}`}
              >
                {hex(b, 2)}
              </span>
            ))}
          </span>
          <span className="x-hex-text">
            {bytes.map((b, j) => {
              const ch = printable(b);
              return (
                <span
                  key={j}
                  data-at={offset + j}
                  className={`${ch === null ? 'is-dot' : ''}${hovered === offset + j ? ' is-hovered' : ''}`}
                >
                  {ch ?? '.'}
                </span>
              );
            })}
          </span>
        </div>
      ))}
      {truncated && (
        <p className="vx-label">
          {(Math.ceil(data.length / BYTES_PER_ROW) - MAX_ROWS).toLocaleString('en-US')} more rows
          not shown ({formatBytes(data.length)} in all).
        </p>
      )}
    </div>
  );
}
