// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useMemo } from 'react';
import type { LayoutTreeNode, SegmentMapEntry } from '../swimlane/types';
import { collectSubtreeSegments, findPathToNode, getNodeDisplayName } from '../swimlane/utils';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { useSelection } from '../../contexts/SelectionContext';
import { DataTable, type CellRenderer } from '../DataTable';

interface SegmentsPaneProps {
  node: LayoutTreeNode;
  segments: SegmentMapEntry[];
}

export function SegmentsPane({ node, segments }: SegmentsPaneProps) {
  const file = useVortexFile();
  const { selectNode, selectSegment, hoverSegment } = useSelection();
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

  const pathRenderer: CellRenderer = useCallback(
    (_value: unknown, row: Record<string, unknown>) => {
      const layoutPath = row.layout_path as string;
      const pathNodes = findPathToNode(file.layoutTree, layoutPath);
      return (
        <span className="text-vortex-grey-dark">
          {pathNodes.map((pathNode, i) => (
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
        </span>
      );
    },
    [file.layoutTree, selectNode],
  );

  const handleRowClick = useCallback(
    (rowIndex: number) => {
      const row = rows[rowIndex];
      if (row) selectSegment(row.index as number);
    },
    [rows, selectSegment],
  );

  const handleRowHover = useCallback(
    (rowIndex: number | null) => {
      if (rowIndex == null) {
        hoverSegment(null);
      } else {
        const row = rows[rowIndex];
        if (row) hoverSegment(row.index as number);
      }
    },
    [rows, hoverSegment],
  );

  if (rows.length === 0) {
    return <div className="text-xs text-vortex-grey-dark p-2.5">No segments for this node.</div>;
  }

  return (
    <div className="h-full">
      <DataTable
        columns={columns}
        rows={rows}
        onRowClick={handleRowClick}
        onRowHover={handleRowHover}
        cellRenderers={{ layout_path: pathRenderer }}
      />
    </div>
  );
}
