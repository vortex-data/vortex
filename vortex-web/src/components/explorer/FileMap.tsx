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
 * Three rendering tiers:
 *  1. Selected segment (if any): bright blue, full opacity.
 *  2. Other subtree segments: dim blue.
 *  3. Everything else: neutral base.
 */
export function FileMap() {
  const file = useVortexFile();
  const { state: selection } = useSelection();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [crosshair, setCrosshair] = useState<number | null>(null);

  // Segments belonging to the selected subtree, sorted by byte offset
  const subtreeSegments = useMemo((): SegmentMapEntry[] => {
    if (!selection.selectedNode) return [];
    const segIds = new Set(collectSubtreeSegments(selection.selectedNode));
    return file.segments
      .filter((s) => segIds.has(s.index))
      .sort((a, b) => a.byteOffset - b.byteOffset);
  }, [selection.selectedNode, file.segments]);

  // The single selected segment (if any), sorted for scan
  const focusedSegment = useMemo((): SegmentMapEntry | null => {
    if (selection.selectedSegmentIndex == null) return null;
    return file.segments.find((s) => s.index === selection.selectedSegmentIndex) ?? null;
  }, [selection.selectedSegmentIndex, file.segments]);

  // Segments for hover preview (from hovered node or hovered segment)
  const hoverSegments = useMemo((): SegmentMapEntry[] => {
    if (selection.hoveredNode) {
      const segIds = new Set(collectSubtreeSegments(selection.hoveredNode));
      return file.segments
        .filter((s) => segIds.has(s.index))
        .sort((a, b) => a.byteOffset - b.byteOffset);
    }
    if (selection.hoveredSegmentIndex != null) {
      const seg = file.segments.find((s) => s.index === selection.hoveredSegmentIndex);
      return seg ? [seg] : [];
    }
    return [];
  }, [selection.hoveredNode, selection.hoveredSegmentIndex, file.segments]);

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

    const dark =
      document.documentElement.classList.contains('dark') ||
      window.matchMedia('(prefers-color-scheme: dark)').matches;

    // Base: neutral
    const baseR = dark ? 50 : 210;
    const baseG = dark ? 50 : 210;
    const baseB = dark ? 50 : 210;
    const baseA = dark ? 80 : 100;

    // Highlight color (vortex light blue #2CB9D1)
    const hlR = 44,
      hlG = 185,
      hlB = 209;

    const hasSubtree = subtreeSegments.length > 0;
    const hasHover = hoverSegments.length > 0;
    const bytesPerPixel = fileSize / width;

    let segIdx = 0;
    let hoverIdx = 0;

    for (let px = 0; px < width; px++) {
      const byteStart = px * bytesPerPixel;
      const byteEnd = (px + 1) * bytesPerPixel;

      // Advance subtree scan
      while (
        segIdx < subtreeSegments.length &&
        subtreeSegments[segIdx].byteOffset + subtreeSegments[segIdx].byteLength <= byteStart
      ) {
        segIdx++;
      }

      // Advance hover scan
      while (
        hoverIdx < hoverSegments.length &&
        hoverSegments[hoverIdx].byteOffset + hoverSegments[hoverIdx].byteLength <= byteStart
      ) {
        hoverIdx++;
      }

      // Check subtree hit
      let isSubtree = false;
      if (hasSubtree) {
        for (let s = segIdx; s < subtreeSegments.length; s++) {
          if (subtreeSegments[s].byteOffset >= byteEnd) break;
          isSubtree = true;
          break;
        }
      }

      // Check hover hit
      let isHovered = false;
      if (hasHover) {
        for (let h = hoverIdx; h < hoverSegments.length; h++) {
          if (hoverSegments[h].byteOffset >= byteEnd) break;
          isHovered = true;
          break;
        }
      }

      // Check focused segment hit
      let isFocused = false;
      if (focusedSegment) {
        const fStart = focusedSegment.byteOffset;
        const fEnd = fStart + focusedSegment.byteLength;
        if (fStart < byteEnd && fEnd > byteStart) {
          isFocused = true;
        }
      }

      let r: number, g: number, b: number, a: number;
      if (isHovered) {
        // Hover always wins — bright highlight
        r = hlR;
        g = hlG;
        b = hlB;
        a = 255;
      } else if (isFocused) {
        r = hlR;
        g = hlG;
        b = hlB;
        a = 255;
      } else if (isSubtree && !focusedSegment) {
        // No segment selected: all subtree segments are bright
        r = hlR;
        g = hlG;
        b = hlB;
        a = dark ? 140 : 160;
      } else if (isSubtree) {
        // A segment is selected: other subtree segments are dimmed
        r = hlR;
        g = hlG;
        b = hlB;
        a = dark ? 90 : 110;
      } else {
        r = baseR;
        g = baseG;
        b = baseB;
        a = baseA;
      }

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
  }, [file, subtreeSegments, focusedSegment, hoverSegments]);

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
