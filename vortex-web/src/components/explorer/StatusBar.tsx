// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContextCore';
import { useSelection } from '../../contexts/SelectionContextCore';
import {
  formatBytes,
  formatRowCount,
  collectSubtreeSegments,
  getNodeDisplayName,
} from '../swimlane/utils';

/** The status line: where the file is read, the file's totals, then the hovered or selected node's. */
export function StatusBar() {
  const file = useVortexFile();
  const { state: selection } = useSelection();

  const activeNode = selection.hoveredNode ?? selection.selectedNode;

  const selectionStats = useMemo(() => {
    if (!activeNode) return null;
    const segIds = new Set(collectSubtreeSegments(activeNode));
    const reachable = file.segments.filter((s) => segIds.has(s.index));
    const bytes = reachable.reduce((sum, s) => sum + s.byteLength, 0);
    const pct = file.fileStructure.fileSize > 0 ? (bytes / file.fileStructure.fileSize) * 100 : 0;
    return { rows: activeNode.rowCount, segments: reachable.length, bytes, pct };
  }, [activeNode, file.segments, file.fileStructure.fileSize]);

  return (
    <footer className="vx-status-line vx-label x-status">
      <span className="vx-status-item is-ok">Read in your browser · nothing uploaded</span>
      <span className="vx-num">
        {formatRowCount(file.rowCount)} rows · {formatBytes(file.fileSize)} · {file.segments.length}{' '}
        segments
      </span>
      {activeNode && selectionStats && (
        <span className="vx-num x-push">
          {getNodeDisplayName(activeNode)} · {formatRowCount(selectionStats.rows)} rows ·{' '}
          {selectionStats.segments} segments · {formatBytes(selectionStats.bytes)} (
          {selectionStats.pct.toFixed(1)}%)
        </span>
      )}
    </footer>
  );
}
