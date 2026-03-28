// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { LayoutTreeNode } from '../swimlane/types';
import type { VortexFileState } from '../../contexts/VortexFileContext';
import {
  formatBytes,
  formatRowCount,
  getNodeDisplayName,
  collectSubtreeSegments,
  shortEncoding,
} from '../swimlane/utils';

function formatNum(n: number): string {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, '_');
}

interface SummaryPaneProps {
  node: LayoutTreeNode | null;
  file: VortexFileState;
}

export function SummaryPane({ node, file }: SummaryPaneProps) {
  if (!node) {
    // File-level summary
    return (
      <div className="text-xs space-y-2">
        <h3 className="font-medium text-vortex-fg-light dark:text-vortex-fg">File Summary</h3>
        <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-vortex-grey-dark">
          <span>File</span>
          <span className="text-vortex-fg-light dark:text-vortex-fg">{file.fileName}</span>
          <span>Size</span>
          <span className="text-vortex-fg-light dark:text-vortex-fg">
            {formatBytes(file.fileSize)}
          </span>
          <span>Rows</span>
          <span className="text-vortex-fg-light dark:text-vortex-fg">
            {formatRowCount(file.rowCount)}
          </span>
          <span>Segments</span>
          <span className="text-vortex-fg-light dark:text-vortex-fg">{file.segments.length}</span>
          <span>Version</span>
          <span className="text-vortex-fg-light dark:text-vortex-fg">v{file.version}</span>
        </div>
      </div>
    );
  }

  // Node-level summary
  const name = getNodeDisplayName(node);
  const subtreeSegmentIds = collectSubtreeSegments(node);
  const reachableSegments = file.segments.filter((s) => subtreeSegmentIds.includes(s.index));
  const totalBytes = reachableSegments.reduce((sum, s) => sum + s.byteLength, 0);

  return (
    <div className="text-xs space-y-2">
      <h3 className="font-medium text-vortex-fg-light dark:text-vortex-fg">{name}</h3>
      <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-vortex-grey-dark">
        <span>Encoding</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg" title={node.encoding}>
          {shortEncoding(node.encoding)}
        </span>
        <span>Rows</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">{formatNum(node.rowCount)}</span>
        <span>Row offset</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">
          {formatNum(node.rowOffset)}
        </span>
        <span>Metadata</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">
          {formatBytes(node.metadataBytes)}
        </span>
        <span>Data size</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">{formatBytes(totalBytes)}</span>
        <span>Segments</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">{reachableSegments.length}</span>
        <span>Children</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">{node.children.length}</span>
      </div>
    </div>
  );
}
