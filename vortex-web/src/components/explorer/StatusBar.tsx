// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { useSelection } from '../../contexts/SelectionContext';
import { formatBytes, formatRowCount, collectSubtreeSegments } from '../swimlane/utils';

export function StatusBar() {
  const file = useVortexFile();
  const { state: selection } = useSelection();

  // Show hovered node stats if hovering, otherwise selected node stats.
  const activeNode = selection.hoveredNode ?? selection.selectedNode;

  const selectionStats = useMemo(() => {
    if (!activeNode) return null;

    const segIds = new Set(collectSubtreeSegments(activeNode));
    const reachable = file.segments.filter((s) => segIds.has(s.index));
    const totalBytes = reachable.reduce((sum, s) => sum + s.byteLength, 0);
    const pct =
      file.fileStructure.fileSize > 0 ? (totalBytes / file.fileStructure.fileSize) * 100 : 0;

    return {
      rows: activeNode.rowCount,
      segments: reachable.length,
      bytes: totalBytes,
      pct,
    };
  }, [activeNode, file.segments, file.fileStructure.fileSize]);

  return (
    <div className="flex items-center px-3 h-6 flex-shrink-0 border-t border-vortex-grey-light/60 dark:border-white/[0.08] bg-vortex-grey-lightest/50 dark:bg-white/[0.02] text-[10px] text-vortex-grey-dark">
      {/* File-level stats — left */}
      <div className="flex items-center gap-3">
        <Stat label="Rows" value={formatRowCount(file.rowCount)} />
        <Stat label="Size" value={formatBytes(file.fileSize)} />
        <Stat label="Segments" value={String(file.segments.length)} />
      </div>

      <div className="flex-1" />

      {/* Selection stats — right */}
      {selectionStats && (
        <div className="flex items-center gap-3">
          <Stat label="rows" value={formatRowCount(selectionStats.rows)} />
          <Stat label="segments" value={String(selectionStats.segments)} />
          <Stat
            label="bytes"
            value={`${formatBytes(selectionStats.bytes)} (${selectionStats.pct.toFixed(1)}%)`}
          />
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="whitespace-nowrap">
      <span className="mr-0.5">{label}:</span>
      <span className="text-vortex-fg-light dark:text-vortex-fg">{value}</span>
    </span>
  );
}
