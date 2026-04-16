// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { FlattenedRow } from './types';
import {
  ROW_HEIGHT,
  DTYPE_COLORS,
  getEncodingStyle,
  getNodeDisplayName,
  hasExpandableChildren,
  formatRowRange,
  getDtypeCategory,
  shortEncoding,
} from './utils';

interface TreeRowProps {
  row: FlattenedRow;
  isExpanded: boolean;
  isSelected: boolean;
  isHovered?: boolean;
  isHoveredAncestor?: boolean;
  mode: 'schema' | 'layout';
  onToggle: () => void;
  onSelect: () => void;
  onHover?: (nodeId: string | null) => void;
}

export function TreeRow({
  row,
  isExpanded,
  isSelected,
  isHovered,
  isHoveredAncestor,
  mode,
  onToggle,
  onSelect,
  onHover,
}: TreeRowProps) {
  const { node, depth, displayKind } = row;

  if (displayKind === 'hiddenIndicator') {
    const name = node.childType.kind === 'transparent' ? node.childType.name : 'hidden';
    return (
      <div
        className="flex items-center text-[10px] text-vortex-grey-dark italic"
        style={{ height: ROW_HEIGHT, paddingLeft: 6 + depth * 10 }}
      >
        <span className="ml-3">{name}</span>
      </div>
    );
  }

  const isGroup = displayKind === 'group';
  const expandable = hasExpandableChildren(node);
  const isLeaf = !expandable;
  const style = getEncodingStyle(node.encoding);
  const name = getNodeDisplayName(node);

  let badgeText: string;
  let badgeColor: string;
  if (isGroup) {
    badgeText = '···';
    badgeColor = style.color;
  } else if (mode === 'schema') {
    const dtypeCat = getDtypeCategory(node.dtype);
    const dtypeStr = shortEncoding(node.dtype);
    badgeText =
      dtypeCat === 'struct'
        ? 'struct'
        : dtypeStr.length > 20
          ? dtypeStr.slice(0, 18) + '…'
          : dtypeStr;
    badgeColor = DTYPE_COLORS[dtypeCat];
  } else {
    const ct = node.childType;
    if (ct.kind === 'chunk') {
      badgeText = formatRowRange(row.rowRange);
    } else {
      badgeText = style.label;
      if (node.children.length > 0 && node.childType.kind !== 'field') {
        badgeText += ` (${node.children.length})`;
      }
    }
    badgeColor = style.color;
  }

  const opacity = isGroup ? 'opacity-50' : isLeaf ? 'opacity-70' : '';
  const fontStyle = isGroup ? 'italic' : '';
  const highlightBg = isSelected
    ? 'bg-vortex-light-blue/10'
    : isHovered
      ? 'bg-vortex-light-blue/15'
      : isHoveredAncestor
        ? 'bg-vortex-light-blue/5'
        : '';

  return (
    <div
      data-node-id={node.id}
      className={`flex items-center gap-1.5 text-[11px] whitespace-nowrap hover:bg-vortex-black/[0.03] dark:hover:bg-white/[0.04] cursor-default ${opacity} ${fontStyle} ${highlightBg}`}
      title={node.isArrayNode ? 'Array encoding node' : undefined}
      style={{
        height: ROW_HEIGHT,
        paddingLeft: 6 + depth * 10,
        paddingRight: 8,
        backgroundColor: node.isArrayNode ? 'var(--array-node-bg)' : undefined,
      }}
      onClick={onSelect}
      onMouseEnter={() => onHover?.(node.id)}
      onMouseLeave={() => onHover?.(null)}
    >
      <span
        className={`text-[8px] w-3 text-vortex-grey-dark ${expandable ? 'cursor-pointer' : ''}`}
        style={{ opacity: expandable ? 1 : 0 }}
        onClick={(e) => {
          e.stopPropagation();
          if (expandable) onToggle();
        }}
      >
        {isExpanded ? '▼' : '▶'}
      </span>
      <span className="flex-1 overflow-hidden text-ellipsis text-vortex-fg-light dark:text-vortex-fg">
        {name}
      </span>
      <span
        className="text-[9px] px-1.5 py-0.5 rounded max-w-[120px] truncate"
        style={{ color: badgeColor, backgroundColor: `${badgeColor}15` }}
      >
        {badgeText}
      </span>
    </div>
  );
}
