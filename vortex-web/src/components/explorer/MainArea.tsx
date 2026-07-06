// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useRef, useState } from 'react';
import { TreePanel } from './TreePanel';
import { SwimlaneOverview } from '../swimlane/SwimlaneOverview';
import { DataPreview } from './DataPreview';
import { DetailPanel } from '../detail/DetailPanel';
import { SummarySidebar } from '../detail/SummarySidebar';

const MIN_PANEL_HEIGHT = 120;
const DEFAULT_PREVIEW_HEIGHT = 200;

export type MainView = 'details' | 'swimlane';

/**
 * Main explorer area: tree panel (left) | the active main view (details or
 * swimlane, chosen in the top header) over a resizable data preview (right). The
 * tree stays put as column navigation while the header tabs switch the view.
 *
 * The preview panel at the bottom is vertically resizable via a drag handle.
 */
export function MainArea({ view }: { view: MainView }) {
  const [previewHeight, setPreviewHeight] = useState(DEFAULT_PREVIEW_HEIGHT);
  const dragging = useRef(false);
  const startY = useRef(0);
  const startHeight = useRef(0);
  const containerRef = useRef<HTMLDivElement>(null);

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
    const containerHeight = containerRef.current?.clientHeight ?? 600;
    const maxPreview = containerHeight - MIN_PANEL_HEIGHT;
    const delta = startY.current - e.clientY;
    const next = Math.min(maxPreview, Math.max(MIN_PANEL_HEIGHT, startHeight.current + delta));
    setPreviewHeight(next);
  }, []);

  const onPointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      {/* Left: tree panel — column navigation paired with the details view. The
          swimlane view hides it so the byte-bars get the full width. */}
      {view === 'details' && (
        <div className="w-[260px] flex-shrink-0 h-full overflow-hidden">
          <TreePanel />
        </div>
      )}

      {/* Right: active main view over the resizable data preview */}
      <div ref={containerRef} className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Active view — fills available vertical space, scrolls internally. The
            swimlane keeps the summary sidebar alongside it, like the details view. */}
        <div className="flex-1 min-h-0 flex overflow-hidden">
          {view === 'details' ? (
            <DetailPanel />
          ) : (
            <>
              <div className="flex-1 min-w-0 flex flex-col">
                <SwimlaneOverview />
              </div>
              <SummarySidebar />
            </>
          )}
        </div>

        {/* Resize handle */}
        <div
          className="flex-shrink-0 h-1 cursor-row-resize border-t border-vortex-grey-light/40 dark:border-white/[0.06] hover:bg-vortex-light-blue/20 active:bg-vortex-light-blue/30 transition-colors"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        />

        {/* Data preview — resizable bottom section */}
        <div className="flex-shrink-0 overflow-hidden" style={{ height: previewHeight }}>
          <DataPreview />
        </div>
      </div>
    </div>
  );
}
