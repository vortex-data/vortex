// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type React from 'react';
import type { LayoutTreeNode, FlattenedRow, PhysicalSegment } from './types';
import { collectSubtreeSegments, dtypeClass, getNodeDisplayName } from './utils';

interface SwimlaneBarProps {
  row: FlattenedRow;
  /** Total byte span of the file — the x-axis maximum. */
  totalBytes: number;
  /** Lookup from segment index to its physical byte placement. */
  segmentIndex: Map<number, PhysicalSegment>;
  onHover: (node: LayoutTreeNode | null, position: { x: number; y: number }) => void;
  /** Focus this node's column when its bar is clicked. */
  onSelect: () => void;
}

const pct = (n: number, total: number) => `${(n / total) * 100}%`;

/**
 * A swimlane bar plotted by physical byte offset. Every segment reachable from the node is drawn
 * at its real position in the file, in its dtype's colour, so a column whose storage is split
 * across the file shows gaps. Each contiguous run sits in a frame you can click; the frame takes
 * the accent when the column is selected (from the lane's `is-selected`). The lane's name is the
 * keyboard path, so the frames are not tab stops.
 */
export function SwimlaneBar({
  row,
  totalBytes,
  segmentIndex,
  onHover,
  onSelect,
}: SwimlaneBarProps) {
  const { node } = row;
  if (totalBytes <= 0) return null;

  const segments = collectSubtreeSegments(node)
    .map((id) => segmentIndex.get(id))
    .filter((s): s is PhysicalSegment => s != null)
    .sort((a, b) => a.byteOffset - b.byteOffset);
  if (segments.length === 0) return null;

  // Merge into contiguous runs. Gaps between runs are physical gaps and stay empty.
  const runs: Array<[number, number]> = [];
  for (const seg of segments) {
    const end = seg.byteOffset + seg.byteLength;
    const last = runs[runs.length - 1];
    if (last && seg.byteOffset <= last[1]) last[1] = Math.max(last[1], end);
    else runs.push([seg.byteOffset, end]);
  }

  const hover = (e: React.PointerEvent) => onHover(node, { x: e.clientX, y: e.clientY });
  const leave = () => onHover(null, { x: 0, y: 0 });
  const name = getNodeDisplayName(node);

  return (
    <>
      {segments.map((seg) => (
        <span
          key={seg.byteOffset}
          className={`x-seg ${dtypeClass(seg.dtype)}`}
          style={{ left: pct(seg.byteOffset, totalBytes), width: pct(seg.byteLength, totalBytes) }}
        />
      ))}
      {runs.map(([start, end], i) => (
        <button
          type="button"
          key={`run-${start}`}
          className="x-run"
          tabIndex={-1}
          aria-label={runs.length > 1 ? `${name}, part ${i + 1} of ${runs.length}` : name}
          style={{ left: pct(start, totalBytes), width: pct(end - start, totalBytes) }}
          onPointerEnter={hover}
          onPointerMove={hover}
          onPointerLeave={leave}
          onClick={onSelect}
        />
      ))}
    </>
  );
}
