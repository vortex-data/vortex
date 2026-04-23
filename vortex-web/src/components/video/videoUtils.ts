// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type {
  DecodeSampleInfo,
  VideoFrameInfo,
  VideoGopInfo,
  VideoPlanningFrameInfo,
  VideoTrackInfo,
} from '../swimlane/types';

export interface ByteRange {
  startByteOffset: number;
  byteLength: number;
}

export const FRAME_TYPE_COLORS: Record<string, string> = {
  I: '#5971FD',
  P: '#CEE562',
  B: '#FB863D',
  Unknown: '#71717A',
};

export function getFrameTypeColor(frameType: string): string {
  return FRAME_TYPE_COLORS[frameType] ?? FRAME_TYPE_COLORS.Unknown;
}

export function frameStartSeconds(frame: VideoFrameInfo, timescale: number): number {
  return timescale > 0 ? frame.pts / timescale : 0;
}

export function frameEndSeconds(frame: VideoFrameInfo, timescale: number): number {
  return timescale > 0
    ? (frame.pts + frame.duration) / timescale
    : frameStartSeconds(frame, timescale);
}

export function frameSeekSeconds(frame: VideoFrameInfo, timescale: number): number {
  const start = frameStartSeconds(frame, timescale);
  const end = frameEndSeconds(frame, timescale);
  return start + Math.max(0, (end - start) * 0.05);
}

export function findFrameForTime(
  frames: VideoFrameInfo[],
  timescale: number,
  timeSeconds: number,
): VideoFrameInfo | null {
  if (frames.length === 0) return null;

  const targetPts = Math.round(timeSeconds * timescale);
  for (const frame of frames) {
    const start = frame.pts;
    const end = frame.pts + frame.duration;
    if (targetPts >= start && targetPts < end) return frame;
  }

  if (timeSeconds <= frameStartSeconds(frames[0], timescale)) return frames[0];
  return frames[frames.length - 1];
}

export function formatDurationMs(durationMs: number): string {
  const seconds = durationMs / 1000;
  return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`;
}

export function formatPts(value: number, timescale: number): string {
  if (timescale <= 0) return String(value);
  return `${(value / timescale).toFixed(3)}s`;
}

export function formatFps(fpsNum: number, fpsDen: number): string {
  if (fpsDen <= 0) return `${fpsNum}`;
  const fps = fpsNum / fpsDen;
  return Number.isInteger(fps) ? `${fps}` : fps.toFixed(2);
}

export function effectiveTrackFps(track: VideoTrackInfo): number | null {
  if (track.durationMs <= 0 || track.frameCount <= 0) return null;
  return (track.frameCount * 1000) / track.durationMs;
}

export function displayTrackFps(track: VideoTrackInfo): string {
  const effectiveFps = effectiveTrackFps(track);
  if (effectiveFps != null && Number.isFinite(effectiveFps) && effectiveFps > 0) {
    return Number.isInteger(effectiveFps) ? `${effectiveFps}` : effectiveFps.toFixed(2);
  }
  return formatFps(track.fpsNum, track.fpsDen);
}

export function formatBytes(value: number): string {
  if (value === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const exp = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  const scaled = value / 1024 ** exp;
  return `${scaled.toFixed(scaled >= 10 || exp === 0 ? 0 : 1)} ${units[exp]}`;
}

export function sourceFileNameFromUri(uri: string): string | null {
  try {
    const clean = uri.split('?')[0].split('#')[0];
    const parts = clean.split('/');
    const last = parts[parts.length - 1];
    return last ? decodeURIComponent(last) : null;
  } catch {
    return null;
  }
}

export function describeTrack(track: VideoTrackInfo): string {
  return `Track ${track.trackId} · ${displayTrackFps(track)} fps · ${track.frameCount} frames`;
}

export function findPlanningFrame(
  track: VideoTrackInfo,
  videoFramePos: number | null,
): VideoPlanningFrameInfo | null {
  if (videoFramePos == null) return null;
  return track.planningFrames.find((frame) => frame.videoFramePos === videoFramePos) ?? null;
}

export function findGop(track: VideoTrackInfo, gopPos: number | null): VideoGopInfo | null {
  if (gopPos == null) return null;
  return track.gops.find((gop) => gop.gopPos === gopPos) ?? null;
}

export function byteRangeForSample(sample: DecodeSampleInfo): ByteRange {
  return {
    startByteOffset: sample.sampleByteOffset,
    byteLength: sample.sampleByteLength,
  };
}

export function mergeByteRanges(ranges: ByteRange[]): ByteRange[] {
  if (ranges.length === 0) return [];
  const sorted = [...ranges]
    .filter((range) => range.byteLength > 0)
    .sort((left, right) => left.startByteOffset - right.startByteOffset);
  if (sorted.length === 0) return [];

  const merged: ByteRange[] = [sorted[0]];
  for (const range of sorted.slice(1)) {
    const current = merged[merged.length - 1];
    const currentEnd = current.startByteOffset + current.byteLength;
    const rangeEnd = range.startByteOffset + range.byteLength;
    if (range.startByteOffset <= currentEnd) {
      current.byteLength = Math.max(currentEnd, rangeEnd) - current.startByteOffset;
    } else {
      merged.push({ ...range });
    }
  }
  return merged;
}

export function totalRangeBytes(ranges: ByteRange[]): number {
  return ranges.reduce((sum, range) => sum + range.byteLength, 0);
}

export function trackSampleRanges(track: VideoTrackInfo): ByteRange[] {
  return mergeByteRanges(track.samplesByDecode.map(byteRangeForSample));
}

export function gopSampleRanges(track: VideoTrackInfo, gop: VideoGopInfo): ByteRange[] {
  const gopDecodePositions = new Set(gop.frames.map((frame) => frame.globalDecodePos));
  return mergeByteRanges(
    track.samplesByDecode
      .filter((sample) => gopDecodePositions.has(sample.globalDecodePos))
      .map(byteRangeForSample),
  );
}

export function decodeClosurePositions(
  track: VideoTrackInfo,
  planningFrame: VideoPlanningFrameInfo,
): number[] {
  const gop = findGop(track, planningFrame.gopPos);
  const gopStartDecodePos =
    gop?.startGlobalDecodePos ??
    track.samplesByDecode.find((sample) => sample.gopPos === planningFrame.gopPos)
      ?.globalDecodePos ??
    planningFrame.globalDecodePos;
  const gopEndDecodePos = gop?.endGlobalDecodePos ?? gopStartDecodePos;

  const positions: number[] = [];
  planningFrame.closureLocalDecodeMaskLe.forEach((maskByte, byteIdx) => {
    for (let bitIdx = 0; bitIdx < 8; bitIdx += 1) {
      if ((maskByte & (1 << bitIdx)) === 0) continue;
      const decodePos = gopStartDecodePos + byteIdx * 8 + bitIdx;
      if (decodePos <= gopEndDecodePos) positions.push(decodePos);
    }
  });

  positions.push(...planningFrame.closureExternalDecodePositions);
  return Array.from(new Set(positions)).sort((left, right) => left - right);
}

export function frameClosureRanges(
  track: VideoTrackInfo,
  planningFrame: VideoPlanningFrameInfo,
): ByteRange[] {
  return mergeByteRanges(
    decodeClosurePositions(track, planningFrame)
      .map((decodePos) => track.samplesByDecode[decodePos])
      .filter((sample): sample is DecodeSampleInfo => sample != null)
      .map(byteRangeForSample),
  );
}
