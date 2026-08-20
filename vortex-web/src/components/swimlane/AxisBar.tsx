// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo } from 'react';
import { formatBytes } from './utils';

interface AxisBarProps {
  totalBytes: number;
  swimlaneMinWidth: number;
  rulerPosition: { x: number; byte: number } | null;
  scrollLeft: number;
  containerWidth: number;
  axisRef: React.RefObject<HTMLDivElement | null>;
}

export function AxisBar({
  totalBytes,
  swimlaneMinWidth,
  rulerPosition,
  scrollLeft,
  containerWidth,
  axisRef,
}: AxisBarProps) {
  const axisTicks = useMemo(() => {
    const ticks = [];
    const step = totalBytes / 5;
    for (let i = 0; i <= 5; i++) {
      ticks.push(Math.round(i * step));
    }
    return ticks;
  }, [totalBytes]);

  return (
    // Reserve the same scrollbar gutter as the swimlane panel so the ticks stay
    // aligned with the bars whether or not the vertical scrollbar is present.
    <div className="flex-1 overflow-hidden relative" style={{ scrollbarGutter: 'stable' }}>
      <div ref={axisRef} className="relative h-[26px]" style={{ minWidth: swimlaneMinWidth }}>
        {axisTicks.map((tick) => (
          <div
            key={tick}
            className="absolute text-[9px] text-vortex-grey-dark top-1.5"
            style={{
              left: `${(tick / totalBytes) * 100}%`,
              transform: 'translateX(-50%)',
            }}
          >
            {formatBytes(tick)}
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
          {formatBytes(rulerPosition.byte)}
        </div>
      )}
    </div>
  );
}
