// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo, useState } from 'react';
import type { LayoutTreeNode, SegmentMapEntry } from '../swimlane/types';
import { formatBytes, collectSubtreeSegments } from '../swimlane/utils';

interface SegmentsPaneProps {
  node: LayoutTreeNode;
  segments: SegmentMapEntry[];
}

type SortKey = 'index' | 'byteOffset' | 'byteLength' | 'alignment';

export function SegmentsPane({ node, segments }: SegmentsPaneProps) {
  const [sortKey, setSortKey] = useState<SortKey>('index');
  const [sortAsc, setSortAsc] = useState(true);

  const subtreeSegmentIds = useMemo(() => new Set(collectSubtreeSegments(node)), [node]);

  const filtered = useMemo(
    () => segments.filter((s) => subtreeSegmentIds.has(s.index)),
    [segments, subtreeSegmentIds],
  );

  const sorted = useMemo(() => {
    const cmp = sortAsc ? 1 : -1;
    return [...filtered].sort((a, b) => (a[sortKey] - b[sortKey]) * cmp);
  }, [filtered, sortKey, sortAsc]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc((prev) => !prev);
    else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const sortIndicator = (key: SortKey) => {
    if (sortKey !== key) return '';
    return sortAsc ? ' ▲' : ' ▼';
  };

  if (sorted.length === 0) {
    return <div className="text-xs text-vortex-grey-dark">No segments for this node.</div>;
  }

  return (
    <div className="text-xs">
      <table className="w-full text-left">
        <thead>
          <tr className="text-vortex-grey-dark border-b border-vortex-grey-lightest dark:border-vortex-grey-dark/30">
            <th
              className="pb-1 pr-3 cursor-pointer hover:text-vortex-black dark:hover:text-vortex-white"
              onClick={() => handleSort('index')}
            >
              #{sortIndicator('index')}
            </th>
            <th
              className="pb-1 pr-3 cursor-pointer hover:text-vortex-black dark:hover:text-vortex-white"
              onClick={() => handleSort('byteOffset')}
            >
              Offset{sortIndicator('byteOffset')}
            </th>
            <th
              className="pb-1 pr-3 cursor-pointer hover:text-vortex-black dark:hover:text-vortex-white"
              onClick={() => handleSort('byteLength')}
            >
              Length{sortIndicator('byteLength')}
            </th>
            <th
              className="pb-1 pr-3 cursor-pointer hover:text-vortex-black dark:hover:text-vortex-white"
              onClick={() => handleSort('alignment')}
            >
              Align{sortIndicator('alignment')}
            </th>
            <th className="pb-1">Path</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((seg) => (
            <tr
              key={seg.index}
              className="border-b border-vortex-grey-lightest/50 dark:border-vortex-grey-dark/10 text-vortex-black dark:text-vortex-white"
            >
              <td className="py-0.5 pr-3 text-vortex-light-blue">{seg.index}</td>
              <td className="py-0.5 pr-3 font-mono text-[10px]">{formatBytes(seg.byteOffset)}</td>
              <td className="py-0.5 pr-3 font-mono text-[10px]">{formatBytes(seg.byteLength)}</td>
              <td className="py-0.5 pr-3">{seg.alignment}</td>
              <td className="py-0.5 font-mono text-[10px] text-vortex-grey-dark truncate max-w-[200px]">
                {seg.layoutPath}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
