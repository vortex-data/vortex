// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { LayoutTreeNode } from '../swimlane/types';
import { formatBytes, getNodeDisplayName, shortEncoding } from '../swimlane/utils';
import { Facts } from './Facts';

interface ArraySummaryPaneProps {
  node: LayoutTreeNode;
}

export function ArraySummaryPane({ node }: ArraySummaryPaneProps) {
  const buffers = node.bufferLengths ?? [];
  const totalBufferBytes = buffers.reduce((s, b) => s + b, 0);

  return (
    <>
      <div className="vx-pane-head">
        <h2 className="vx-label x-side-title">{getNodeDisplayName(node)}</h2>
      </div>
      <div className="vx-pane-body">
        <Facts
          items={[
            ['Encoding', shortEncoding(node.encoding), node.encoding],
            ['Metadata', formatBytes(node.metadataBytes)],
            ['Buffers', String(buffers.length)],
            ['Buffer data', formatBytes(totalBufferBytes)],
            ['Children', String(node.children.length)],
          ]}
        />
      </div>
    </>
  );
}
