// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import React from 'react';
import type { LayoutTreeNode, FlattenedRow } from './types';
import { getDtypeCategory, getNodeRowRange } from './utils';
import { DTYPE_COLORS, getEncodingStyle } from './styles';

interface SwimlaneBarProps {
  row: FlattenedRow;
  totalRows: number;
  onHover: (node: LayoutTreeNode | null, position: { x: number; y: number }) => void;
}

export function SwimlaneBar({ row, totalRows, onHover }: SwimlaneBarProps) {
  const { node, displayKind, groupedChildren } = row;
  const isLeaf = node.children.length === 0;
  const isGroup = displayKind === 'group';
  const style = getEncodingStyle(node.encoding);

  // Group nodes: render rolled-up bars from each grouped child
  if (isGroup && groupedChildren) {
    return (
      <>
        {groupedChildren.map((child) => {
          const range = getNodeRowRange(child);
          const left = (range[0] / totalRows) * 100;
          const width = ((range[1] - range[0]) / totalRows) * 100;
          const dtypeCat = getDtypeCategory(child.dtype);
          const dtypeColor = DTYPE_COLORS[dtypeCat];
          return (
            <div
              key={child.id}
              className="absolute top-[3px] bottom-[3px] rounded"
              style={{
                left: `calc(${left}% + 1px)`,
                width: `calc(${width}% - 3px)`,
                backgroundColor: `${dtypeColor}40`,
                border: `1.5px solid ${dtypeColor}`,
              }}
            />
          );
        })}
      </>
    );
  }

  const rowRange = getNodeRowRange(node);
  const left = (rowRange[0] / totalRows) * 100;
  const width = ((rowRange[1] - rowRange[0]) / totalRows) * 100;

  let barClasses = 'absolute top-[3px] bottom-[3px] rounded transition-[filter] duration-100';
  const barStyle: React.CSSProperties = {
    left: `calc(${left}% + 1px)`,
    width: `calc(${width}% - 3px)`,
  };

  if (isLeaf) {
    const dtypeCat = getDtypeCategory(node.dtype);
    const dtypeColor = DTYPE_COLORS[dtypeCat];
    barStyle.backgroundColor = `${dtypeColor}40`;
    barStyle.border = `1.5px solid ${dtypeColor}`;
    barClasses += ' cursor-pointer';
  } else {
    barStyle.border = `1.5px solid ${style.color}40`;
  }

  const handleMouseEnter = (e: React.MouseEvent) => {
    if (isLeaf) {
      (e.currentTarget as HTMLDivElement).style.filter = 'brightness(1.15)';
      onHover(node, { x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isLeaf) {
      onHover(node, { x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseLeave = (e: React.MouseEvent) => {
    if (isLeaf) {
      (e.currentTarget as HTMLDivElement).style.filter = '';
      onHover(null, { x: 0, y: 0 });
    }
  };

  return (
    <div
      className={barClasses}
      style={barStyle}
      onMouseEnter={handleMouseEnter}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    />
  );
}
