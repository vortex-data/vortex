// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useMemo, useState } from 'react';
import type { LayoutTreeNode, SegmentMapEntry } from '../swimlane/types';
import { formatBytes, collectSubtreeSegments, findPathToNode, getNodeDisplayName } from '../swimlane/utils';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { useSelection } from '../../contexts/SelectionContext';

interface SegmentsPaneProps {
  node: LayoutTreeNode;
  segments: SegmentMapEntry[];
}

type SortKey = 'index' | 'byteOffset' | 'byteLength' | 'alignment';

/** Format a number with thin-space digit grouping (e.g. 1 234 567). */
function spaceGroup(n: number): string {
  const s = String(n);
  const result: string[] = [];
  for (let i = s.length; i > 0; i -= 3) {
    result.unshift(s.slice(Math.max(0, i - 3), i));
  }
  return result.join('\u2009');
}

/** Format a byte offset with space-separated digit groups. */
function formatOffset(bytes: number): string {
  return spaceGroup(bytes);
}

/** Format a relative offset like +1 234 or −5 678. */
function formatRelativeOffset(bytes: number, anchor: number): string {
  const diff = bytes - anchor;
  if (diff === 0) return '0';
  const sign = diff > 0 ? '+' : '\u2212';
  return `${sign}${spaceGroup(Math.abs(diff))}`;
}

export function SegmentsPane({ node, segments }: SegmentsPaneProps) {
  const file = useVortexFile();
  const [sortKey, setSortKey] = useState<SortKey>('index');
  const [sortAsc, setSortAsc] = useState(true);
  const { state: selection, selectSegment, selectNode, hoverNode, hoverSegment } = useSelection();

  const subtreeSegmentIds = useMemo(() => new Set(collectSubtreeSegments(node)), [node]);

  const filtered = useMemo(
    () => segments.filter((s) => subtreeSegmentIds.has(s.index)),
    [segments, subtreeSegmentIds],
  );

  const sorted = useMemo(() => {
    const cmp = sortAsc ? 1 : -1;
    return [...filtered].sort((a, b) => (a[sortKey] - b[sortKey]) * cmp);
  }, [filtered, sortKey, sortAsc]);

  const anchorSegment = useMemo(() => {
    if (selection.selectedSegmentIndex == null) return null;
    return segments.find((s) => s.index === selection.selectedSegmentIndex) ?? null;
  }, [selection.selectedSegmentIndex, segments]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc((prev) => !prev);
    else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const handleRowClick = (segIndex: number) => {
    selectSegment(selection.selectedSegmentIndex === segIndex ? null : segIndex);
  };

  const sortIndicator = (key: SortKey) => {
    if (sortKey !== key) return '';
    return sortAsc ? ' \u25B2' : ' \u25BC';
  };

  if (sorted.length === 0) {
    return <div className="text-xs text-vortex-grey-dark">No segments for this node.</div>;
  }

  return (
    <div className="text-xs px-2.5 pb-2.5">
      <table className="w-full text-left">
        <thead>
          <tr className="text-vortex-grey-dark">
            <th
              className="sticky top-0 w-px whitespace-nowrap pb-1 pr-3 cursor-pointer hover:text-vortex-fg-light dark:hover:text-vortex-fg bg-vortex-white dark:bg-vortex-black border-b border-vortex-grey-light/40 dark:border-white/[0.06]"
              onClick={() => handleSort('index')}
            >
              #{sortIndicator('index')}
            </th>
            <th
              className="sticky top-0 w-px whitespace-nowrap pb-1 pr-3 text-right cursor-pointer hover:text-vortex-fg-light dark:hover:text-vortex-fg bg-vortex-white dark:bg-vortex-black border-b border-vortex-grey-light/40 dark:border-white/[0.06]"
              onClick={() => handleSort('byteOffset')}
            >
              Offset{sortIndicator('byteOffset')}
            </th>
            <th
              className="sticky top-0 w-px whitespace-nowrap pb-1 pr-3 cursor-pointer hover:text-vortex-fg-light dark:hover:text-vortex-fg bg-vortex-white dark:bg-vortex-black border-b border-vortex-grey-light/40 dark:border-white/[0.06]"
              onClick={() => handleSort('byteLength')}
            >
              Length{sortIndicator('byteLength')}
            </th>
            <th
              className="sticky top-0 w-px whitespace-nowrap pb-1 pr-3 cursor-pointer hover:text-vortex-fg-light dark:hover:text-vortex-fg bg-vortex-white dark:bg-vortex-black border-b border-vortex-grey-light/40 dark:border-white/[0.06]"
              onClick={() => handleSort('alignment')}
            >
              Align{sortIndicator('alignment')}
            </th>
            <th className="sticky top-0 pb-1 bg-vortex-white dark:bg-vortex-black border-b border-vortex-grey-light/40 dark:border-white/[0.06]">Path</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((seg) => {
            const isSelected = selection.selectedSegmentIndex === seg.index;
            return (
              <tr
                key={seg.index}
                className={`border-b border-vortex-grey-light/20 dark:border-white/[0.04] cursor-default ${
                  isSelected
                    ? 'bg-vortex-light-blue/10 text-vortex-fg-light dark:text-vortex-fg'
                    : 'text-vortex-fg-light dark:text-vortex-fg hover:bg-vortex-black/[0.02] dark:hover:bg-white/[0.02]'
                }`}
                onClick={() => handleRowClick(seg.index)}
                onMouseEnter={() => hoverSegment(seg.index)}
                onMouseLeave={() => hoverSegment(null)}
              >
                <td className="py-0.5 pr-3 whitespace-nowrap text-vortex-light-blue">{seg.index}</td>
                <td className="py-0.5 pr-3 whitespace-nowrap font-mono text-[10px] text-right tabular-nums" title={formatOffset(seg.byteOffset)}>
                  {isSelected || !anchorSegment
                    ? formatOffset(seg.byteOffset)
                    : formatRelativeOffset(seg.byteOffset, anchorSegment.byteOffset)}
                </td>
                <td className="py-0.5 pr-3 whitespace-nowrap font-mono text-[10px]">{formatBytes(seg.byteLength)}</td>
                <td className="py-0.5 pr-3 whitespace-nowrap">{seg.alignment}</td>
                <td className="py-0.5 font-mono text-[10px] text-vortex-grey-dark truncate max-w-[200px]">
                  {findPathToNode(file.layoutTree, seg.layoutPath).map((pathNode, i) => (
                    <span key={pathNode.id}>
                      {i > 0 && <span className="opacity-40">/</span>}
                      <button
                        className="hover:text-vortex-light-blue"
                        onClick={(e) => {
                          e.stopPropagation();
                          selectNode(pathNode.id);
                        }}
                      >
                        {getNodeDisplayName(pathNode)}
                      </button>
                    </span>
                  ))}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
