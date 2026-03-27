// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { FlattenedRow } from './types';
import {
  ROW_HEIGHT,
  getEncodingStyle,
  getNodeDisplayName,
  hasExpandableChildren,
  formatRowRange,
} from './utils';

interface TreeRowProps {
  row: FlattenedRow;
  isExpanded: boolean;
  isSelected: boolean;
  mode: 'schema' | 'layout';
  onToggle: () => void;
  onSelect: () => void;
}

export function TreeRow({ row, isExpanded, isSelected, mode, onToggle, onSelect }: TreeRowProps) {
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
  if (isGroup) {
    badgeText = '···';
  } else if (mode === 'schema') {
    badgeText = node.dtype.length > 20 ? node.dtype.slice(0, 18) + '…' : node.dtype;
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
  }

  const opacity = isGroup ? 'opacity-50' : isLeaf ? 'opacity-70' : '';
  const fontStyle = isGroup ? 'italic' : '';
  const selectedBg = isSelected ? 'bg-vortex-light-blue/10' : '';

  return (
    <div
      className={`flex items-center gap-1.5 text-[11px] whitespace-nowrap hover:bg-vortex-grey-lightest dark:hover:bg-vortex-grey-dark/20 cursor-default ${opacity} ${fontStyle} ${selectedBg}`}
      style={{ height: ROW_HEIGHT, paddingLeft: 6 + depth * 10, paddingRight: 8 }}
      onClick={onSelect}
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
      <span className="flex-1 overflow-hidden text-ellipsis text-vortex-black dark:text-vortex-white">
        {name}
      </span>
      <span
        className="text-[9px] px-1.5 py-0.5 rounded max-w-[120px] truncate"
        style={{ color: style.color, backgroundColor: `${style.color}15` }}
      >
        {badgeText}
      </span>
    </div>
  );
}
