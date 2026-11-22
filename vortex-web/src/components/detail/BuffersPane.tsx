// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import type { LayoutTreeNode } from '../swimlane/types';
import { formatBytes } from '../swimlane/utils';
import { useVortexFile } from '../../contexts/VortexFileContextCore';
import { BufferSelectionController } from './bufferSelection';

interface BuffersPaneProps {
  node: LayoutTreeNode;
  controller?: BufferSelectionController;
}

export function BuffersPane({ node, controller: externalController }: BuffersPaneProps) {
  const { fetchArrayBuffer } = useVortexFile();
  const bufferLengths = node.bufferLengths ?? [];

  const controllerRef = useRef<BufferSelectionController | null>(null);
  if (!externalController && !controllerRef.current) {
    controllerRef.current = new BufferSelectionController(fetchArrayBuffer);
  }
  const controller = externalController ?? controllerRef.current!;
  controller.setFetchArrayBuffer(fetchArrayBuffer);

  const snapshot = useSyncExternalStore(
    controller.subscribe,
    controller.getSnapshot,
    controller.getSnapshot,
  );
  const isCurrentNode = snapshot.nodeId === node.id;
  const selectedBuffer = isCurrentNode ? snapshot.selectedBuffer : null;
  const bufferData = isCurrentNode ? snapshot.bufferData : null;
  const loading = isCurrentNode ? snapshot.loading : false;
  const error = isCurrentNode ? snapshot.error : null;

  const bufferNames: string[] = bufferLengths.map((_, i) => node.bufferNames?.[i] ?? `buffer ${i}`);

  // Auto-select first buffer when node changes and invalidate stale reads on change/unmount.
  useEffect(() => {
    void controller.selectNode(node.id, bufferLengths.length);
    return () => {
      controller.invalidate();
    };
  }, [controller, node.id, bufferLengths.length, fetchArrayBuffer]);

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
            onClick={() => void controller.selectBuffer(i)}
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
