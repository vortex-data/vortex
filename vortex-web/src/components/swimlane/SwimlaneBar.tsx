// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import React from 'react';
import type { LayoutTreeNode, FlattenedRow, PhysicalSegment } from './types';
import { collectSubtreeSegments, getDtypeCategory, MIN_CELL_WIDTH } from './utils';
import { DTYPE_COLORS, getEncodingStyle } from './styles';

interface SwimlaneBarProps {
  row: FlattenedRow;
  /** Total byte span of the file — the x-axis maximum. */
  totalBytes: number;
  /** Lookup from segment index to its physical byte placement. */
  segmentIndex: Map<number, PhysicalSegment>;
  onHover: (node: LayoutTreeNode | null, position: { x: number; y: number }) => void;
  /** Focus this node's column when its bar is clicked. */
  onSelect: () => void;
  /** Whether this node's column is the current focused selection. */
  isSelected: boolean;
}

/**
 * A swimlane bar plotted by physical byte offset. Every segment reachable from
 * the node is drawn at its real position in the file, so a column whose storage
 * is split across the file (interleaved with other columns) shows visible gaps.
 */
export function SwimlaneBar({
  row,
  totalBytes,
  segmentIndex,
  onHover,
  onSelect,
  isSelected,
}: SwimlaneBarProps) {
  const { node } = row;
  const style = getEncodingStyle(node.encoding);

  if (totalBytes <= 0) return null;

  // Physical segments reachable from this node, in file order.
  const segments = collectSubtreeSegments(node)
    .map((id) => segmentIndex.get(id))
    .filter((s): s is PhysicalSegment => s != null)
    .sort((a, b) => a.byteOffset - b.byteOffset);

  if (segments.length === 0) return null;

  // Merge into contiguous runs for the encoding frame. Gaps between runs are
  // physical gaps in the column's representation and are left empty.
  const runs: Array<[number, number]> = [];
  for (const seg of segments) {
    const start = seg.byteOffset;
    const end = seg.byteOffset + seg.byteLength;
    const last = runs[runs.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else runs.push([start, end]);
  }

  const handleEnter = (e: React.MouseEvent) => onHover(node, { x: e.clientX, y: e.clientY });
  const handleMove = (e: React.MouseEvent) => onHover(node, { x: e.clientX, y: e.clientY });
  const handleLeave = () => onHover(null, { x: 0, y: 0 });

  return (
    <>
      {segments.map((seg) => {
        const color = DTYPE_COLORS[getDtypeCategory(seg.dtype)];
        return (
          <div
            key={seg.byteOffset}
            className="absolute top-[3px] bottom-[3px]"
            style={{
              left: `${(seg.byteOffset / totalBytes) * 100}%`,
              width: `${(seg.byteLength / totalBytes) * 100}%`,
              minWidth: MIN_CELL_WIDTH,
              border: `1px solid ${color}`,
              boxSizing: 'border-box',
            }}
          />
        );
      })}
      {runs.map(([start, end]) => (
        <div
          key={`run-${start}`}
          className="absolute top-[3px] bottom-[3px] rounded cursor-pointer"
          style={{
            left: `calc(${(start / totalBytes) * 100}% + 1px)`,
            width: `calc(${((end - start) / totalBytes) * 100}% - 3px)`,
            minWidth: MIN_CELL_WIDTH,
            border: `1.5px solid ${style.color}${isSelected ? '' : '40'}`,
            backgroundColor: isSelected ? `${style.color}33` : undefined,
          }}
          onMouseEnter={handleEnter}
          onMouseMove={handleMove}
          onMouseLeave={handleLeave}
          onClick={onSelect}
        />
      ))}
    </>
  );
}
