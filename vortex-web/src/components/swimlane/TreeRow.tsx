// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { CSSProperties } from 'react';
import type { FlattenedRow } from './types';
import { Icon } from '../Icon';
import {
  dtypeClass,
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
  /** The node's data size, for the right column. */
  size?: string;
  /** 0 for the one node in the tab order, -1 for the rest. */
  tabIndex?: number;
  onToggle: () => void;
  onSelect: () => void;
  onHover?: (nodeId: string | null) => void;
}

/** What the type column says: the dtype in Schema mode, the layout or row range in Layout mode. */
function typeText(row: FlattenedRow, mode: 'schema' | 'layout'): string {
  const { node, displayKind } = row;
  if (displayKind === 'group') return '···';
  if (mode === 'schema') {
    if (getDtypeCategory(node.dtype) === 'struct') return 'struct';
    const d = shortEncoding(node.dtype);
    return d.length > 24 ? `${d.slice(0, 22)}…` : d;
  }
  if (node.childType.kind === 'chunk') return formatRowRange(row.rowRange);
  const label = getEncodingStyle(node.encoding).label;
  return node.children.length > 0 && node.childType.kind !== 'field'
    ? `${label} · ${node.children.length}`
    : label;
}

/**
 * One node of the layout tree: a design system `vx-tree-node` with the tree pattern's roles. The
 * parent tree (`role="tree"`) owns the keys; a click selects, and a click on the chevron opens or
 * closes the node.
 */
export function TreeRow({
  row,
  isExpanded,
  isSelected,
  isHovered,
  isHoveredAncestor,
  mode,
  size,
  tabIndex = -1,
  onToggle,
  onSelect,
  onHover,
}: TreeRowProps) {
  const { node, depth, displayKind } = row;
  const depthStyle = { '--vx-depth': depth } as CSSProperties;

  if (displayKind === 'hiddenIndicator') {
    const name = node.childType.kind === 'transparent' ? node.childType.name : 'hidden';
    return (
      <div
        className="vx-tree-node is-hidden"
        role="treeitem"
        aria-level={depth + 1}
        aria-selected={false}
        aria-disabled="true"
        tabIndex={tabIndex}
        data-node-id={node.id}
        style={depthStyle}
      >
        <span className="vx-tree-name x-tree-name">
          <Icon name="chevron_right" className="vx-tree-toggle is-leaf" />
          <span className="x-tree-label">{name}</span>
        </span>
      </div>
    );
  }

  const expandable = hasExpandableChildren(node);
  const classes = [
    'vx-tree-node',
    displayKind === 'group' && 'is-group',
    node.isArrayNode && 'is-array',
    isHovered && !isSelected && 'is-hovered',
    isHoveredAncestor && !isSelected && 'is-hovered-parent',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    // The tree (role="tree") takes the keys for all its nodes, as the WAI-ARIA tree pattern does.
    // biome-ignore lint/a11y/useKeyWithClickEvents: the tree handles the keys.
    <div
      className={classes}
      role="treeitem"
      aria-level={depth + 1}
      aria-selected={isSelected}
      {...(expandable ? { 'aria-expanded': isExpanded } : {})}
      tabIndex={tabIndex}
      data-node-id={node.id}
      title={node.isArrayNode ? 'Array encoding node' : node.dtype}
      style={depthStyle}
      onClick={(e) => {
        // A click on the chevron opens or closes the node and leaves the selection as it is.
        if (expandable && (e.target as Element).closest('.vx-tree-toggle')) onToggle();
        else onSelect();
      }}
      onMouseEnter={() => onHover?.(node.id)}
      onMouseLeave={() => onHover?.(null)}
    >
      <span className="vx-tree-name x-tree-name">
        <Icon name="chevron_right" className={`vx-tree-toggle${expandable ? '' : ' is-leaf'}`} />
        <span className={`vx-tree-dot ${dtypeClass(node.dtype)}`} />
        <span className="x-tree-label">{getNodeDisplayName(node)}</span>
        <span className="vx-tree-type">{typeText(row, mode)}</span>
      </span>
      {size && <span className="vx-tree-size">{size}</span>}
    </div>
  );
}
