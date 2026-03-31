// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo } from 'react';
import { formatRowCount } from './utils';

interface AxisBarProps {
  totalRows: number;
  swimlaneMinWidth: number;
  rulerPosition: { x: number; row: number } | null;
  scrollLeft: number;
  containerWidth: number;
  axisRef: React.RefObject<HTMLDivElement | null>;
}

export function AxisBar({
  totalRows,
  swimlaneMinWidth,
  rulerPosition,
  scrollLeft,
  containerWidth,
  axisRef,
}: AxisBarProps) {
  const axisTicks = useMemo(() => {
    const ticks = [];
    const step = totalRows / 5;
    for (let i = 0; i <= 5; i++) {
      ticks.push(Math.round(i * step));
    }
    return ticks;
  }, [totalRows]);

  return (
    <div className="flex-1 overflow-hidden relative">
      <div ref={axisRef} className="relative h-[26px]" style={{ minWidth: swimlaneMinWidth }}>
        {axisTicks.map((tick) => (
          <div
            key={tick}
            className="absolute text-[9px] text-vortex-grey-dark top-1.5"
            style={{
              left: `${(tick / totalRows) * 100}%`,
              transform: 'translateX(-50%)',
            }}
          >
            {formatRowCount(tick)}
          </div>
        ))}
      </div>

      {rulerPosition && (
        <div
          className="absolute top-1 bg-vortex-black dark:bg-vortex-white text-vortex-white dark:text-vortex-black px-1.5 py-0.5 rounded text-[10px] font-medium pointer-events-none z-[100] whitespace-nowrap"
          style={{
            left: Math.max(0, Math.min(rulerPosition.x - scrollLeft - 20, containerWidth - 50)),
          }}
        >
          {formatRowCount(rulerPosition.row)}
        </div>
      )}
    </div>
  );
}
