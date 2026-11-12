// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { Facts } from '../detail/Facts';
import type { LayoutTreeNode } from './types';
import {
  dtypeClass,
  formatBytes,
  getNodeDisplayName,
  getNodeRowRange,
  shortEncoding,
} from './utils';

interface TooltipProps {
  node: LayoutTreeNode;
  position: { x: number; y: number };
}

/** The card that follows the pointer over a swimlane bar. */
export function Tooltip({ node, position }: TooltipProps) {
  const [start, end] = getNodeRowRange(node);
  const items: [string, string, string?][] = [['Rows', (end - start).toLocaleString('en-US')]];
  if (node.dtype) items.push(['Dtype', node.dtype]);
  if (node.encoding) items.push(['Encoding', shortEncoding(node.encoding), node.encoding]);
  if (node.metadataBytes > 0) items.push(['Metadata', formatBytes(node.metadataBytes)]);

  return (
    <div className="x-tip" style={{ left: position.x + 12, top: position.y - 10 }}>
      <p className={`vx-mono x-tip-head x-dtype ${dtypeClass(node.dtype)}`}>
        {getNodeDisplayName(node)}
      </p>
      <Facts items={items} />
    </div>
  );
}
