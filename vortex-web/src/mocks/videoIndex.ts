// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type {
  DecodeSampleInfo,
  VideoFrameInfo,
  VideoGopInfo,
  VideoIndexInfo,
  VideoPlanningFrameInfo,
  VideoTrackInfo,
} from '../components/swimlane/types';

function makeTrack(
  trackId: number,
  fpsNum: number,
  fpsDen: number,
  videoFramePositions: number[],
  offsetBase: number,
): VideoTrackInfo {
  const framesByGop = new Map<number, number[]>();
  videoFramePositions.forEach((videoFramePos) => {
    const gopPos = videoFramePos < 6 ? 0 : 1;
    const frames = framesByGop.get(gopPos) ?? [];
    frames.push(videoFramePos);
    framesByGop.set(gopPos, frames);
  });

  let runningOffset = offsetBase;
  let globalDecodePos = 0;
  const gops: VideoGopInfo[] = [];
  const planningFrames: VideoPlanningFrameInfo[] = [];
  const samplesByDecode: DecodeSampleInfo[] = [];

  for (const [gopPos, positions] of [...framesByGop.entries()].sort((left, right) => left[0] - right[0])) {
    const frames: VideoFrameInfo[] = positions.map((videoFramePos, gopIndex) => {
      const sampleByteLength = 80_000 + ((videoFramePos + trackId) % 4) * 14_000;
      const frameType =
        gopIndex === 0 ? 'I' : gopIndex % 3 === 0 ? 'P' : gopIndex % 2 === 0 ? 'B' : 'P';
      const frame: VideoFrameInfo = {
        sampleId: globalDecodePos + 1,
        globalDecodePos,
        videoFramePos,
        gopPos,
        gopFramePos: gopIndex,
        pts: videoFramePos * 1000,
        dts: videoFramePos * 1000,
        duration:
          positions[gopIndex + 1] != null
            ? (positions[gopIndex + 1] - videoFramePos) * 1000
            : Math.round((fpsDen / fpsNum) * 30_000),
        displayPos: gopIndex,
        decodePos: gopIndex,
        frameType,
        isSync: gopIndex === 0,
        frameNum: videoFramePos,
        isReference: frameType !== 'B',
        sampleByteOffset: runningOffset,
        sampleByteLength,
        refL0DecodePositions: gopIndex > 0 ? [Math.max(0, gopIndex - 1)] : [],
        refL1DecodePositions: frameType === 'B' && gopIndex + 1 < positions.length ? [gopIndex + 1] : [],
        refL0GlobalDecodePositions: gopIndex > 0 ? [Math.max(0, globalDecodePos - 1)] : [],
        refL1GlobalDecodePositions:
          frameType === 'B' && gopIndex + 1 < positions.length ? [globalDecodePos + 1] : [],
        refPrevDecodePos: gopIndex > 0 ? globalDecodePos - 1 : null,
        refNextDecodePos:
          frameType === 'B' && gopIndex + 1 < positions.length ? globalDecodePos + 1 : null,
        dependencyDepth: gopIndex === 0 ? 0 : frameType === 'B' ? 2 : 1,
      };

      samplesByDecode.push({
        sampleId: frame.sampleId,
        globalDecodePos: frame.globalDecodePos,
        videoFramePos: frame.videoFramePos,
        gopPos: frame.gopPos,
        gopFramePos: frame.gopFramePos,
        gopDecodePos: frame.decodePos,
        pts: frame.pts,
        dts: frame.dts,
        duration: frame.duration,
        sampleByteOffset: frame.sampleByteOffset,
        sampleByteLength: frame.sampleByteLength,
        isSync: frame.isSync,
      });

      planningFrames.push({
        videoFramePos: frame.videoFramePos,
        globalDecodePos: frame.globalDecodePos,
        gopPos: frame.gopPos,
        gopDecodePos: frame.decodePos,
        sampleByteOffset: frame.sampleByteOffset,
        sampleByteLength: frame.sampleByteLength,
        closureLocalDecodeMaskLe: [Math.max(1, (1 << (gopIndex + 1)) - 1)],
        closureExternalDecodePositions: [],
      });

      runningOffset += sampleByteLength + 12_000;
      globalDecodePos += 1;
      return frame;
    });

    const first = frames[0];
    const last = frames[frames.length - 1];
    gops.push({
      gopPos,
      startPts: first.pts,
      endPts: last.pts + last.duration,
      startDts: first.dts,
      endDts: last.dts + last.duration,
      startByteOffset: first.sampleByteOffset,
      byteLength: last.sampleByteOffset + last.sampleByteLength - first.sampleByteOffset,
      frameCount: frames.length,
      keyframeDecodePos: 0,
      dependencyTreeHeight: 2,
      startGlobalDecodePos: frames[0].globalDecodePos,
      endGlobalDecodePos: frames[frames.length - 1].globalDecodePos,
      frames,
    });
  }

  const frames = gops.flatMap((gop) => gop.frames);
  return {
    trackId,
    trackLanguage: 'und',
    width: 1280,
    height: 720,
    fpsNum,
    fpsDen,
    timescale: 30_000,
    durationTs: 12_000,
    durationMs: 400,
    nalLengthSize: 4,
    frameCount: frames.length,
    gops,
    frames,
    planningFrames,
    samplesByDecode,
  };
}

export function makeVideoIndexMock(): VideoIndexInfo {
  const tracks: VideoTrackInfo[] = [
    makeTrack(1, 5, 1, [0, 6], 60_000),
    makeTrack(2, 15, 2, [0, 4, 8], 80_000),
    makeTrack(3, 10, 1, [0, 3, 6, 9], 120_000),
    makeTrack(4, 15, 1, [0, 2, 4, 6, 8, 10], 180_000),
    makeTrack(5, 30, 1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 260_000),
  ];

  const primaryTrack = tracks[tracks.length - 1];

  return {
    sourceUri: '/tmp/mock-video.mp4',
    container: 'mp4',
    codec: 'h264',
    primaryTrackId: primaryTrack.trackId,
    trackLanguage: primaryTrack.trackLanguage,
    width: primaryTrack.width,
    height: primaryTrack.height,
    fpsNum: primaryTrack.fpsNum,
    fpsDen: primaryTrack.fpsDen,
    timescale: primaryTrack.timescale,
    durationTs: primaryTrack.durationTs,
    durationMs: primaryTrack.durationMs,
    fileSizeBytes: 2_600_000,
    nalLengthSize: primaryTrack.nalLengthSize,
    frameCount: primaryTrack.frameCount,
    gops: primaryTrack.gops,
    frames: primaryTrack.frames,
    planningFrames: primaryTrack.planningFrames,
    samplesByDecode: primaryTrack.samplesByDecode,
    tracks,
  };
}
