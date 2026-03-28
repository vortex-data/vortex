// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { LayoutTreeNode } from '../swimlane/types';
import { formatBytes, getNodeDisplayName, shortEncoding } from '../swimlane/utils';

interface ArraySummaryPaneProps {
  node: LayoutTreeNode;
}

export function ArraySummaryPane({ node }: ArraySummaryPaneProps) {
  const name = getNodeDisplayName(node);
  const totalBufferBytes = (node.bufferLengths ?? []).reduce((s, b) => s + b, 0);

  return (
    <div className="text-xs space-y-2">
      <h3 className="font-medium text-vortex-fg-light dark:text-vortex-fg">{name}</h3>
      <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-vortex-grey-dark">
        <span>Encoding</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg" title={node.encoding}>
          {shortEncoding(node.encoding)}
        </span>
        <span>Metadata</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">
          {formatBytes(node.metadataBytes)}
        </span>
        <span>Buffers</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">
          {(node.bufferLengths ?? []).length}
        </span>
        <span>Buffer data</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">
          {formatBytes(totalBufferBytes)}
        </span>
        <span>Children</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">{node.children.length}</span>
      </div>
    </div>
  );
}
