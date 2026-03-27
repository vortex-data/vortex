// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { useSelection } from '../../contexts/SelectionContext';
import {
  formatBytes,
  formatRowCount,
  collectSubtreeSegments,
} from '../swimlane/utils';

export function StatusBar() {
  const file = useVortexFile();
  const { state: selection } = useSelection();

  const selectionStats = useMemo(() => {
    const node = selection.selectedNode;
    if (!node) return null;

    const segIds = new Set(collectSubtreeSegments(node));
    const reachable = file.segments.filter((s) => segIds.has(s.index));
    const totalBytes = reachable.reduce((sum, s) => sum + s.byteLength, 0);
    const pct = file.fileStructure.fileSize > 0
      ? (totalBytes / file.fileStructure.fileSize) * 100
      : 0;

    return {
      rows: node.rowCount,
      segments: reachable.length,
      bytes: totalBytes,
      pct,
    };
  }, [selection.selectedNode, file.segments, file.fileStructure.fileSize]);

  return (
    <div className="flex items-center px-3 h-6 flex-shrink-0 border-t border-vortex-grey-light dark:border-vortex-grey-dark bg-vortex-grey-lightest/50 dark:bg-vortex-grey-dark/20 text-[10px] text-vortex-grey-dark">
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
      <span className="text-vortex-black dark:text-vortex-white">{value}</span>
    </span>
  );
}
