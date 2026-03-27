// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { LayoutTreeNode } from '../swimlane/types';
import { getNodeDisplayName } from '../swimlane/utils';

interface EncodingPaneProps {
  node: LayoutTreeNode;
}

/**
 * Displays the encoding tree for a selected node.
 * Currently shows a text representation; will display WASM `display_tree()` output later.
 */
export function EncodingPane({ node }: EncodingPaneProps) {
  return (
    <div className="text-xs">
      <h3 className="font-medium text-vortex-black dark:text-vortex-white mb-2">Encoding Tree</h3>
      <pre className="font-mono text-[10px] text-vortex-grey-dark bg-vortex-grey-lightest dark:bg-vortex-black/50 rounded p-2 overflow-auto">
        {renderTree(node, 0)}
      </pre>
    </div>
  );
}

function renderTree(node: LayoutTreeNode, indent: number): string {
  const prefix = '  '.repeat(indent);
  const name = getNodeDisplayName(node);
  const line = `${prefix}${name}: ${node.encoding} (${node.dtype}, ${node.rowCount} rows)`;

  if (node.children.length === 0) return line;

  const childLines = node.children.map((child) => renderTree(child, indent + 1));
  return [line, ...childLines].join('\n');
}
