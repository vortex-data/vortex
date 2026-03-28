// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { LayoutTreeNode } from './types';
import {
  getDtypeCategory,
  formatBytes,
  getNodeDisplayName,
  getNodeRowRange,
  shortEncoding,
} from './utils';
import { DTYPE_COLORS } from './styles';

interface TooltipProps {
  node: LayoutTreeNode;
  position: { x: number; y: number };
}

export function Tooltip({ node, position }: TooltipProps) {
  const rowRange = getNodeRowRange(node);
  const rows = rowRange[1] - rowRange[0];
  const dtypeCat = getDtypeCategory(node.dtype);
  const dtypeColor = DTYPE_COLORS[dtypeCat];
  const name = getNodeDisplayName(node);

  return (
    <div
      className="fixed z-[1000] pointer-events-none max-w-[220px] rounded-lg border border-vortex-grey-light/60 dark:border-white/[0.1] bg-vortex-white dark:bg-[#252528] shadow-lg p-2 text-xs"
      style={{ left: position.x + 12, top: position.y - 10 }}
    >
      <div className="flex items-center gap-1.5 mb-1">
        <span className="font-medium text-vortex-fg-light dark:text-vortex-fg">{name}</span>
        <span
          className="text-[9px] px-1.5 py-0.5 rounded"
          style={{ color: dtypeColor, backgroundColor: `${dtypeColor}20` }}
        >
          {dtypeCat}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-vortex-grey-dark">
        <span>rows</span>
        <span className="text-vortex-fg-light dark:text-vortex-fg">{rows.toLocaleString()}</span>
        {node.dtype && (
          <>
            <span>dtype</span>
            <span className="text-vortex-fg-light dark:text-vortex-fg">{node.dtype}</span>
          </>
        )}
        {node.encoding && (
          <>
            <span>encoding</span>
            <span className="text-vortex-fg-light dark:text-vortex-fg" title={node.encoding}>
              {shortEncoding(node.encoding)}
            </span>
          </>
        )}
        {node.metadataBytes > 0 && (
          <>
            <span>metadata</span>
            <span className="text-vortex-fg-light dark:text-vortex-fg">
              {formatBytes(node.metadataBytes)}
            </span>
          </>
        )}
      </div>
    </div>
  );
}
