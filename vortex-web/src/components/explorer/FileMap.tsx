// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import { collectSubtreeSegments } from '../swimlane/utils';
import { useVortexFile } from '../../contexts/VortexFileContext';
import { useSelection } from '../../contexts/SelectionContext';
import type { SegmentMapEntry } from '../swimlane/types';

/**
 * Pixel-level byte map of the file. Height matches one line of text (1lh).
 *
 * - No selection: uniform dimmed bar.
 * - With selection: highlights the byte ranges belonging to the selected
 *   node's subtree segments.
 */
export function FileMap() {
  const file = useVortexFile();
  const { state: selection } = useSelection();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [crosshair, setCrosshair] = useState<number | null>(null);

  // Segments belonging to the selected subtree, sorted by byte offset
  const selectedSegments = useMemo((): SegmentMapEntry[] => {
    if (!selection.selectedNode) return [];
    const segIds = new Set(collectSubtreeSegments(selection.selectedNode));
    return file.segments
      .filter((s) => segIds.has(s.index))
      .sort((a, b) => a.byteOffset - b.byteOffset);
  }, [selection.selectedNode, file.segments]);

  // Paint
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width === 0 || height === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const fileSize = file.fileStructure.fileSize;
    if (fileSize === 0) return;

    const imgData = ctx.createImageData(width * dpr, height * dpr);
    const data = imgData.data;
    const imgW = imgData.width;
    const imgH = imgData.height;

    const dark = document.documentElement.classList.contains('dark') ||
      window.matchMedia('(prefers-color-scheme: dark)').matches;

    const baseR = dark ? 50 : 210;
    const baseG = dark ? 50 : 210;
    const baseB = dark ? 50 : 210;
    const baseA = dark ? 80 : 100;

    // Highlight color (vortex light blue #2CB9D1)
    const hlR = 44, hlG = 185, hlB = 209;

    const hasSelection = selectedSegments.length > 0;
    const bytesPerPixel = fileSize / width;

    let segIdx = 0;

    for (let px = 0; px < width; px++) {
      const byteStart = px * bytesPerPixel;
      const byteEnd = (px + 1) * bytesPerPixel;

      while (
        segIdx < selectedSegments.length &&
        selectedSegments[segIdx].byteOffset + selectedSegments[segIdx].byteLength <= byteStart
      ) {
        segIdx++;
      }

      let isHighlighted = false;
      if (hasSelection) {
        for (let s = segIdx; s < selectedSegments.length; s++) {
          if (selectedSegments[s].byteOffset >= byteEnd) break;
          isHighlighted = true;
          break;
        }
      }

      const r = isHighlighted ? hlR : baseR;
      const g = isHighlighted ? hlG : baseG;
      const b = isHighlighted ? hlB : baseB;
      const a = isHighlighted ? 255 : baseA;

      const dprPxStart = Math.round(px * dpr);
      const dprPxEnd = Math.round((px + 1) * dpr);

      for (let dpx = dprPxStart; dpx < dprPxEnd && dpx < imgW; dpx++) {
        for (let dy = 0; dy < imgH; dy++) {
          const idx = (dy * imgW + dpx) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
          data[idx + 3] = a;
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [file, selectedSegments]);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      const canvas = canvasRef.current;
      if (canvas) canvas.dispatchEvent(new Event('resize'));
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    setCrosshair(e.clientX - rect.left);
  }, []);

  const handleMouseLeave = useCallback(() => setCrosshair(null), []);

  return (
    <div
      ref={containerRef}
      className="relative cursor-crosshair flex-shrink-0 h-[1lh] text-[10px] leading-none"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <canvas ref={canvasRef} className="block" />
      {crosshair !== null && (
        <div
          className="absolute top-0 bottom-0 w-px bg-vortex-black dark:bg-vortex-white opacity-50 pointer-events-none"
          style={{ left: crosshair }}
        />
      )}
    </div>
  );
}
