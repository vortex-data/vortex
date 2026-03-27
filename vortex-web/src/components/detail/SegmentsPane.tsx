// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo } from 'react';
import type { LayoutTreeNode, SegmentMapEntry } from '../swimlane/types';
import { collectSubtreeSegments } from '../swimlane/utils';
import { DataTable } from '../DataTable';

interface SegmentsPaneProps {
  node: LayoutTreeNode;
  segments: SegmentMapEntry[];
}

export function SegmentsPane({ node, segments }: SegmentsPaneProps) {
  const subtreeSegmentIds = useMemo(() => new Set(collectSubtreeSegments(node)), [node]);

  const { columns, rows } = useMemo(() => {
    const filtered = segments.filter((s) => subtreeSegmentIds.has(s.index));
    const cols = ['index', 'byte_offset', 'byte_length', 'alignment', 'column', 'layout_path'];
    const rowData = filtered.map((s) => ({
      index: s.index,
      byte_offset: s.byteOffset,
      byte_length: s.byteLength,
      alignment: s.alignment,
      column: s.column ?? '',
      layout_path: s.layoutPath,
    }));
    return { columns: cols, rows: rowData };
  }, [segments, subtreeSegmentIds]);

  if (rows.length === 0) {
    return (
      <div className="text-xs text-vortex-grey-dark p-2.5">
        No segments for this node.
      </div>
    );
  }

  return (
    <div className="h-full">
      <DataTable columns={columns} rows={rows} />
    </div>
  );
}
