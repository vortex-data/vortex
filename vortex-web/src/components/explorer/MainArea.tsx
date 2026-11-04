// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useRef, useState, type CSSProperties } from 'react';
import { TreePanel } from './TreePanel';
import { SwimlaneOverview } from '../swimlane/SwimlaneOverview';
import { DataPreview } from './DataPreview';
import { DetailPanel } from '../detail/DetailPanel';
import { SummarySidebar } from '../detail/SummarySidebar';

const MIN_PANEL_HEIGHT = 120;
const DEFAULT_PREVIEW_HEIGHT = 240;
const KEY_STEP = 24;

export type MainView = 'details' | 'swimlane';

/**
 * The work area: the layout tree (Details view only), then the active view and the summary over
 * the data preview. The handle between them sets the preview's height, by drag or with the arrow
 * keys.
 */
export function MainArea({ view }: { view: MainView }) {
  const [previewHeight, setPreviewHeight] = useState(DEFAULT_PREVIEW_HEIGHT);
  const dragging = useRef(false);
  const startY = useRef(0);
  const startHeight = useRef(0);
  const mainRef = useRef<HTMLDivElement>(null);

  // The view and the preview share the room the other rows leave (the handle, and below 1100px
  // the summary), so the preview grows only until the view is down to its minimum.
  const clamp = useCallback((h: number) => {
    const view = mainRef.current?.firstElementChild;
    const preview = mainRef.current?.lastElementChild;
    const shared =
      view && preview
        ? view.getBoundingClientRect().height + preview.getBoundingClientRect().height
        : 600;
    const max = Math.max(MIN_PANEL_HEIGHT, shared - MIN_PANEL_HEIGHT);
    return Math.round(Math.min(max, Math.max(MIN_PANEL_HEIGHT, h)));
  }, []);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      dragging.current = true;
      startY.current = e.clientY;
      startHeight.current = previewHeight;
      e.currentTarget.setPointerCapture(e.pointerId);
    },
    [previewHeight],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging.current) return;
      setPreviewHeight(clamp(startHeight.current + startY.current - e.clientY));
    },
    [clamp],
  );

  const onPointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const step = { ArrowUp: KEY_STEP, ArrowDown: -KEY_STEP }[e.key];
      if (step === undefined) return;
      e.preventDefault();
      setPreviewHeight((h) => clamp(h + step));
    },
    [clamp],
  );

  return (
    <main className={`x-work${view === 'details' ? '' : ' is-wide'}`}>
      {view === 'details' && <TreePanel />}
      <div
        ref={mainRef}
        className="x-main"
        style={{ '--x-preview-height': `${previewHeight}px` } as CSSProperties}
      >
        {view === 'details' ? <DetailPanel /> : <SwimlaneOverview />}
        <SummarySidebar />
        {/* A window splitter: the focusable separator pattern of WAI-ARIA. */}
        {/* biome-ignore lint/a11y/useSemanticElements: an hr cannot take focus or a drag. */}
        <div
          className="x-handle"
          role="separator"
          aria-orientation="horizontal"
          aria-label="Preview height"
          aria-valuenow={previewHeight}
          aria-valuemin={MIN_PANEL_HEIGHT}
          tabIndex={0}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onKeyDown={onKeyDown}
        />
        <DataPreview />
      </div>
    </main>
  );
}
