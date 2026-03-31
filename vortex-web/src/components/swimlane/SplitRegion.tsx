// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import React, { useState } from 'react';
import type { Split } from './types';
import { MIN_LABEL_WIDTH } from './styles';

interface SplitRegionProps {
  split: Split;
  totalRows: number;
  swimlaneWidth: number;
  isSelected: boolean;
  onClick: (e: React.MouseEvent) => void;
}

export function SplitRegion({
  split,
  totalRows,
  swimlaneWidth,
  isSelected,
  onClick,
}: SplitRegionProps) {
  const [isHovered, setIsHovered] = useState(false);

  const left = (split.rowRange[0] / totalRows) * 100;
  const width = ((split.rowRange[1] - split.rowRange[0]) / totalRows) * 100;
  const widthPx = ((split.rowRange[1] - split.rowRange[0]) / totalRows) * swimlaneWidth;
  const showLabel = widthPx >= MIN_LABEL_WIDTH || isSelected || isHovered;

  return (
    <div
      className="absolute top-0 bottom-0 cursor-pointer border-r border-vortex-grey-light/40 dark:border-white/[0.06]"
      style={{
        left: `${left}%`,
        width: `${width}%`,
        backgroundColor: isSelected
          ? 'rgba(44, 185, 209, 0.08)'
          : isHovered
            ? 'rgba(44, 185, 209, 0.03)'
            : undefined,
      }}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        className="absolute top-1 left-1/2 -translate-x-1/2 text-[9px] px-1.5 py-0.5 rounded whitespace-nowrap transition-opacity duration-150 pointer-events-none"
        style={{
          color: isSelected ? 'rgba(44, 185, 209, 1)' : 'rgba(143, 143, 143, 1)',
          backgroundColor: isSelected ? 'rgba(44, 185, 209, 0.15)' : 'rgba(241, 241, 241, 0.9)',
          opacity: showLabel ? 1 : 0,
          fontWeight: isSelected ? 500 : 400,
        }}
      >
        {split.id}
      </div>
    </div>
  );
}
