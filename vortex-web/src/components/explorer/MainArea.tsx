// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useRef, useState } from 'react';
import { TreePanel } from './TreePanel';
import { FileMap } from './FileMap';
import { DataPreview } from './DataPreview';
import { DetailPanel } from '../detail/DetailPanel';

const MIN_PREVIEW_HEIGHT = 80;
const MAX_PREVIEW_HEIGHT = 600;
const DEFAULT_PREVIEW_HEIGHT = 200;

/**
 * Main explorer area: tree panel (left) | detail + filemap + preview (right).
 *
 * The preview panel at the bottom is vertically resizable via a drag handle.
 */
export function MainArea() {
  const [previewHeight, setPreviewHeight] = useState(DEFAULT_PREVIEW_HEIGHT);
  const dragging = useRef(false);
  const startY = useRef(0);
  const startHeight = useRef(0);

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      dragging.current = true;
      startY.current = e.clientY;
      startHeight.current = previewHeight;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [previewHeight],
  );

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    const delta = startY.current - e.clientY;
    const next = Math.min(MAX_PREVIEW_HEIGHT, Math.max(MIN_PREVIEW_HEIGHT, startHeight.current + delta));
    setPreviewHeight(next);
  }, []);

  const onPointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      {/* Left: tree panel — full height, fixed width */}
      <div className="w-[260px] flex-shrink-0 h-full overflow-hidden">
        <TreePanel />
      </div>

      {/* Right: detail pane, file map, data preview — stacked vertically */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Detail pane — fills available vertical space, scrolls internally */}
        <DetailPanel />

        {/* File map strip */}
        <div className="flex-shrink-0">
          <FileMap />
        </div>

        {/* Resize handle */}
        <div
          className="flex-shrink-0 h-1 cursor-row-resize border-t border-vortex-grey-light/40 dark:border-white/[0.06] hover:bg-vortex-light-blue/20 active:bg-vortex-light-blue/30 transition-colors"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        />

        {/* Data preview — resizable bottom section */}
        <div
          className="flex-shrink-0 overflow-hidden"
          style={{ height: previewHeight }}
        >
          <DataPreview />
        </div>
      </div>
    </div>
  );
}
