// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type MouseEvent,
  type ReactNode,
} from 'react';
import { TreePanel } from '../explorer/TreePanel';
import { useVortexFile } from '../../contexts/VortexFileContext';
import type { VideoFrameInfo, VideoGopInfo, VideoIndexInfo, VideoTrackInfo } from '../swimlane/types';
import { VideoInspector } from './VideoInspector';
import {
  describeTrack,
  findFrameForTime,
  formatDurationMs,
  formatPts,
  frameSeekSeconds,
  getFrameTypeColor,
  sourceFileNameFromUri,
} from './videoUtils';

interface VideoExplorerProps {
  onAttachMedia: (file: File) => void;
}

export function VideoExplorer({ onAttachMedia }: VideoExplorerProps) {
  const file = useVortexFile();
  const videoIndex = file.videoIndex!;
  const videoRef = useRef<HTMLVideoElement>(null);
  const attachInputRef = useRef<HTMLInputElement>(null);
  const selectedFrameRef = useRef<VideoFrameInfo | null>(null);
  const currentTimeRef = useRef(0);

  const tracks = useMemo(
    () =>
      videoIndex.tracks.length > 0 ? videoIndex.tracks : [compatibilityTrack(videoIndex)],
    [videoIndex],
  );
  const defaultTrackId =
    tracks.find((track) => track.trackId === videoIndex.primaryTrackId)?.trackId ?? tracks[0]?.trackId ?? 0;
  const trackById = useMemo(() => new Map(tracks.map((track) => [track.trackId, track])), [tracks]);

  const [selectedTrackId, setSelectedTrackId] = useState<number>(defaultTrackId);
  const [hoveredTrackId, setHoveredTrackId] = useState<number | null>(null);
  const [selectedFramePos, setSelectedFramePos] = useState<number>(0);
  const [hoveredFramePos, setHoveredFramePos] = useState<number | null>(null);
  const [selectedGopPos, setSelectedGopPos] = useState<number | null>(null);
  const [hoveredGopPos, setHoveredGopPos] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    setSelectedTrackId(defaultTrackId);
    setSelectedFramePos(0);
    setSelectedGopPos(null);
    setHoveredTrackId(null);
    setHoveredFramePos(null);
    setHoveredGopPos(null);
    setCurrentTime(0);
    setIsPlaying(false);
  }, [defaultTrackId, videoIndex.sourceUri]);

  const selectedTrack = trackById.get(selectedTrackId) ?? tracks[0] ?? null;
  const orderedFrames = useMemo(
    () =>
      selectedTrack
        ? [...selectedTrack.frames].sort((left, right) => left.videoFramePos - right.videoFramePos)
        : [],
    [selectedTrack],
  );
  const frameByPos = useMemo(
    () => new Map(orderedFrames.map((frame) => [frame.videoFramePos, frame])),
    [orderedFrames],
  );
  const gopByPos = useMemo(
    () => new Map((selectedTrack?.gops ?? []).map((gop) => [gop.gopPos, gop])),
    [selectedTrack],
  );

  useEffect(() => {
    currentTimeRef.current = currentTime;
  }, [currentTime]);

  useEffect(() => {
    if (!selectedTrack) return;
    const nextFrame =
      findFrameForTime(orderedFrames, selectedTrack.timescale, currentTimeRef.current) ??
      orderedFrames[0] ??
      null;
    setSelectedFramePos(nextFrame?.videoFramePos ?? 0);
    setSelectedGopPos(nextFrame?.gopPos ?? selectedTrack.gops[0]?.gopPos ?? null);
    setHoveredFramePos(null);
    setHoveredGopPos(null);
  }, [orderedFrames, selectedTrack, selectedTrackId]);

  const selectedFrame = frameByPos.get(selectedFramePos) ?? orderedFrames[0] ?? null;
  const hoveredFrame = hoveredFramePos != null ? frameByPos.get(hoveredFramePos) ?? null : null;
  const selectedGop =
    (selectedGopPos != null ? gopByPos.get(selectedGopPos) ?? null : null) ??
    (selectedFrame ? gopByPos.get(selectedFrame.gopPos) ?? null : null);
  const hoveredGop = hoveredGopPos != null ? gopByPos.get(hoveredGopPos) ?? null : null;
  const currentFrame = useMemo(
    () =>
      selectedTrack
        ? findFrameForTime(orderedFrames, selectedTrack.timescale, currentTime)
        : null,
    [currentTime, orderedFrames, selectedTrack],
  );

  useEffect(() => {
    selectedFrameRef.current = selectedFrame;
  }, [selectedFrame]);

  useEffect(() => {
    const frame = selectedFrameRef.current;
    if (!selectedTrack || !file.pairedMedia?.hasMedia || !file.pairedMedia.objectUrl || !frame || !videoRef.current) {
      return;
    }

    const nextTime = frameSeekSeconds(frame, selectedTrack.timescale);
    videoRef.current.currentTime = nextTime;
    setCurrentTime(nextTime);
  }, [file.pairedMedia?.hasMedia, file.pairedMedia?.objectUrl, selectedTrackId, selectedTrack]);

  const syncFrameSelection = useCallback((frame: VideoFrameInfo) => {
    setSelectedFramePos(frame.videoFramePos);
    setSelectedGopPos(frame.gopPos);
  }, []);

  const seekToFrame = useCallback(
    (videoFramePos: number) => {
      if (!selectedTrack) return;
      const frame = frameByPos.get(videoFramePos);
      if (!frame) return;
      syncFrameSelection(frame);
      const nextTime = frameSeekSeconds(frame, selectedTrack.timescale);
      setCurrentTime(nextTime);
      const video = videoRef.current;
      if (video && file.pairedMedia?.hasMedia) {
        video.currentTime = nextTime;
      }
    },
    [file.pairedMedia?.hasMedia, frameByPos, selectedTrack, syncFrameSelection],
  );

  const selectFrame = useCallback(
    (videoFramePos: number, seek = false) => {
      const frame = frameByPos.get(videoFramePos);
      if (!frame) return;
      syncFrameSelection(frame);
      if (seek) seekToFrame(videoFramePos);
    },
    [frameByPos, seekToFrame, syncFrameSelection],
  );

  const stepFrame = useCallback(
    (delta: number) => {
      if (!selectedFrame || orderedFrames.length === 0) return;
      const currentIndex = orderedFrames.findIndex(
        (frame) => frame.videoFramePos === selectedFrame.videoFramePos,
      );
      if (currentIndex < 0) return;
      const nextIndex = Math.max(0, Math.min(orderedFrames.length - 1, currentIndex + delta));
      const nextFrame = orderedFrames[nextIndex];
      if (nextFrame) seekToFrame(nextFrame.videoFramePos);
    },
    [orderedFrames, seekToFrame, selectedFrame],
  );

  const jumpGop = useCallback(
    (delta: number) => {
      if (!selectedTrack || !selectedGop) return;
      const currentIndex = selectedTrack.gops.findIndex((gop) => gop.gopPos === selectedGop.gopPos);
      const nextIndex = Math.max(0, Math.min(selectedTrack.gops.length - 1, currentIndex + delta));
      const nextGop = selectedTrack.gops[nextIndex];
      const firstFrame = [...nextGop.frames].sort(
        (left, right) => left.videoFramePos - right.videoFramePos,
      )[0];
      if (firstFrame) seekToFrame(firstFrame.videoFramePos);
    },
    [seekToFrame, selectedGop, selectedTrack],
  );

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video || !selectedTrack) return;
    setCurrentTime(video.currentTime);
    setIsPlaying(!video.paused);
    const frame = findFrameForTime(orderedFrames, selectedTrack.timescale, video.currentTime);
    if (frame) syncFrameSelection(frame);
  }, [orderedFrames, selectedTrack, syncFrameSelection]);

  const togglePlayback = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !file.pairedMedia?.hasMedia) return;
    if (video.paused) {
      await video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  }, [file.pairedMedia?.hasMedia]);

  const handleAttachInput = useCallback(() => attachInputRef.current?.click(), []);
  const handleAttachChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const mediaFile = event.target.files?.[0];
      if (mediaFile) onAttachMedia(mediaFile);
      event.target.value = '';
    },
    [onAttachMedia],
  );

  const handleSelectTrack = useCallback((trackId: number) => {
    setSelectedTrackId(trackId);
    setHoveredTrackId(null);
  }, []);

  const sourceFileName =
    sourceFileNameFromUri(videoIndex.sourceUri) ??
    `${file.fileName.replace(/\.(vortex|vtx)$/i, '')}.mp4`;

  if (!selectedTrack) {
    return null;
  }

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      <div className="w-[260px] flex-shrink-0 h-full overflow-hidden">
        <TreePanel />
      </div>

      <div className="flex-1 min-w-0 flex overflow-hidden">
        <div className="flex-1 min-w-0 flex flex-col overflow-hidden">
          <div className="border-b border-vortex-grey-light/40 dark:border-white/[0.06] px-4 py-3 flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="text-sm font-medium text-vortex-fg-light dark:text-vortex-fg truncate">
                {sourceFileName}
              </div>
              <div className="text-[10px] text-vortex-grey-dark">
                {selectedTrack.width}×{selectedTrack.height} · {videoIndex.codec} · track{' '}
                {selectedTrack.trackId} · {tracks.length} tracks ·{' '}
                {formatDurationMs(selectedTrack.durationMs)}
              </div>
              <div className="mt-2 flex items-center gap-1.5 overflow-x-auto pb-1">
                {tracks.map((track) => {
                  const isSelected = track.trackId === selectedTrack.trackId;
                  const isHovered = track.trackId === hoveredTrackId;
                  return (
                    <button
                      key={track.trackId}
                      className={`rounded-md border px-2.5 py-1 text-[10px] whitespace-nowrap transition-colors ${
                        isSelected
                          ? 'border-vortex-light-blue bg-vortex-light-blue/12 text-vortex-light-blue'
                          : isHovered
                            ? 'border-vortex-light-blue/60 text-vortex-fg-light dark:text-vortex-fg'
                            : 'border-vortex-grey-light/40 dark:border-white/[0.08] text-vortex-grey-dark'
                      }`}
                      onClick={() => handleSelectTrack(track.trackId)}
                      onMouseEnter={() => setHoveredTrackId(track.trackId)}
                      onMouseLeave={() => setHoveredTrackId(null)}
                    >
                      {describeTrack(track)}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="flex items-center gap-1.5 flex-shrink-0">
              <ControlButton onClick={() => jumpGop(-1)}>Prev GOP</ControlButton>
              <ControlButton onClick={() => stepFrame(-1)}>-1</ControlButton>
              <ControlButton onClick={togglePlayback} disabled={!file.pairedMedia?.hasMedia}>
                {isPlaying ? 'Pause' : 'Play'}
              </ControlButton>
              <ControlButton onClick={() => stepFrame(1)}>+1</ControlButton>
              <ControlButton onClick={() => jumpGop(1)}>Next GOP</ControlButton>
            </div>
          </div>

          <div className="flex-1 min-h-0 overflow-auto px-4 py-4 space-y-4">
            <div className="rounded-xl border border-vortex-grey-light/40 dark:border-white/[0.06] overflow-hidden bg-vortex-grey-lightest/40 dark:bg-white/[0.02]">
              <div className="flex items-center justify-between gap-3 border-b border-vortex-grey-light/30 dark:border-white/[0.06] px-4 py-2 text-[10px] text-vortex-grey-dark">
                <div>
                  track {selectedTrack.trackId} · frame {selectedFrame?.videoFramePos ?? '–'} · GOP{' '}
                  {selectedGop?.gopPos ?? '–'} · pts{' '}
                  {selectedFrame ? formatPts(selectedFrame.pts, selectedTrack.timescale) : '–'}
                </div>
                <div>
                  time {currentTime.toFixed(2)}s · decode {selectedFrame?.globalDecodePos ?? '–'}
                </div>
              </div>

              {file.pairedMedia?.hasMedia && file.pairedMedia.objectUrl ? (
                <video
                  ref={videoRef}
                  className="block w-full max-h-[420px] bg-black"
                  src={file.pairedMedia.objectUrl}
                  playsInline
                  onClick={togglePlayback}
                  onTimeUpdate={handleTimeUpdate}
                  onPause={() => setIsPlaying(false)}
                  onPlay={() => setIsPlaying(true)}
                />
              ) : (
                <div className="h-[320px] flex flex-col items-center justify-center gap-3 text-center px-8">
                  <div className="text-sm font-medium text-vortex-fg-light dark:text-vortex-fg">
                    Attach the companion MP4 to enable playback
                  </div>
                  <div className="text-[11px] text-vortex-grey-dark max-w-md">
                    Video-index metadata is loaded, but this persisted `.vortex` file does not
                    contain the source media bytes. Attach <code>{sourceFileName}</code> to scrub
                    the clip alongside the track, GOP, and frame views.
                  </div>
                  <button
                    className="rounded bg-vortex-light-blue px-3 py-1.5 text-sm text-white hover:opacity-90 transition-opacity"
                    onClick={handleAttachInput}
                  >
                    Attach MP4
                  </button>
                </div>
              )}
            </div>

            <TimelineStrip
              track={selectedTrack}
              frames={orderedFrames}
              gops={selectedTrack.gops}
              selectedFramePos={selectedFrame?.videoFramePos ?? null}
              currentFramePos={currentFrame?.videoFramePos ?? null}
              hoveredFramePos={hoveredFramePos}
              hoveredGopPos={hoveredGopPos}
              onSelectFrame={(videoFramePos) => selectFrame(videoFramePos, true)}
              onHoverFrame={setHoveredFramePos}
              onHoverGop={setHoveredGopPos}
            />
          </div>
        </div>

        <VideoInspector
          videoIndex={videoIndex}
          selectedTrack={selectedTrack}
          hoveredTrackId={hoveredTrackId}
          selectedFrame={selectedFrame}
          hoveredFrame={hoveredFrame}
          selectedGop={selectedGop}
          hoveredGop={hoveredGop}
          onSelectTrack={handleSelectTrack}
          onHoverTrack={setHoveredTrackId}
          onSelectFrame={(videoFramePos) => selectFrame(videoFramePos, true)}
          onHoverFrame={setHoveredFramePos}
          onHoverGop={setHoveredGopPos}
        />
      </div>

      <input
        ref={attachInputRef}
        type="file"
        accept=".mp4"
        className="hidden"
        onChange={handleAttachChange}
      />
    </div>
  );
}

function TimelineStrip({
  track,
  frames,
  gops,
  selectedFramePos,
  currentFramePos,
  hoveredFramePos,
  hoveredGopPos,
  onSelectFrame,
  onHoverFrame,
  onHoverGop,
}: {
  track: VideoTrackInfo;
  frames: VideoFrameInfo[];
  gops: VideoGopInfo[];
  selectedFramePos: number | null;
  currentFramePos: number | null;
  hoveredFramePos: number | null;
  hoveredGopPos: number | null;
  onSelectFrame: (videoFramePos: number) => void;
  onHoverFrame: (videoFramePos: number | null) => void;
  onHoverGop: (gopPos: number | null) => void;
}) {
  const width = 1000;
  const height = 170;
  const usableWidth = width - 80;
  const minFramePos = frames[0]?.videoFramePos ?? 0;
  const maxFramePos = frames[frames.length - 1]?.videoFramePos ?? 0;
  const positionToX = (framePos: number) =>
    40 + ((framePos - minFramePos) / Math.max(maxFramePos - minFramePos, 1)) * usableWidth;

  const handleSvgClick = (event: MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    const approxFrame = Math.round(minFramePos + ratio * (maxFramePos - minFramePos));
    const nearest = frames.reduce((best, candidate) => {
      if (!best) return candidate;
      return Math.abs(candidate.videoFramePos - approxFrame) <
        Math.abs(best.videoFramePos - approxFrame)
        ? candidate
        : best;
    }, frames[0]);
    if (nearest) onSelectFrame(nearest.videoFramePos);
  };

  return (
    <div className="rounded-xl border border-vortex-grey-light/40 dark:border-white/[0.06] px-4 py-3 bg-vortex-grey-lightest/30 dark:bg-white/[0.02]">
      <div className="flex items-center justify-between gap-3 mb-2">
        <div>
          <div className="text-sm font-medium text-vortex-fg-light dark:text-vortex-fg">
            Track Timeline
          </div>
          <div className="text-[10px] text-vortex-grey-dark">
            {describeTrack(track)}. GOP bands sit above retained display-order frames.
          </div>
        </div>
        <div className="text-right text-[10px] text-vortex-grey-dark">
          <div>selected frame {selectedFramePos ?? '–'}</div>
          <div>playhead {currentFramePos ?? '–'}</div>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        onClick={handleSvgClick}
        onMouseLeave={() => {
          onHoverFrame(null);
          onHoverGop(null);
        }}
      >
        <text x={14} y={25} className="fill-vortex-grey-dark text-[11px]">
          GOPs
        </text>
        <text x={14} y={95} className="fill-vortex-grey-dark text-[11px]">
          Frames
        </text>

        {gops.map((gop) => {
          const sortedFrames = [...gop.frames].sort(
            (left, right) => left.videoFramePos - right.videoFramePos,
          );
          const firstFrame = sortedFrames[0];
          const lastFrame = sortedFrames[sortedFrames.length - 1];
          if (!firstFrame || !lastFrame) return null;
          const x = positionToX(firstFrame.videoFramePos);
          const bandWidth = Math.max(positionToX(lastFrame.videoFramePos) - x + 6, 10);
          const isActive =
            selectedFramePos != null &&
            selectedFramePos >= firstFrame.videoFramePos &&
            selectedFramePos <= lastFrame.videoFramePos;
          const isHovered = hoveredGopPos === gop.gopPos;

          return (
            <g
              key={gop.gopPos}
              onClick={(event) => event.stopPropagation()}
              onMouseEnter={() => onHoverGop(gop.gopPos)}
              onMouseLeave={() => onHoverGop(null)}
            >
              <rect
                x={x}
                y={34}
                width={bandWidth}
                height={28}
                rx={8}
                fill={isActive ? '#5971FD' : '#2CB9D1'}
                fillOpacity={isHovered ? 0.9 : isActive ? 0.8 : 0.32}
                stroke={isHovered ? '#18181B' : 'transparent'}
                strokeWidth={isHovered ? 2 : 0}
                className="cursor-pointer"
                onClick={() => onSelectFrame(firstFrame.videoFramePos)}
              />
              <text x={x + 8} y={52} className="fill-white text-[10px]">
                GOP {gop.gopPos}
              </text>
            </g>
          );
        })}

        {frames.map((frame) => {
          const x = positionToX(frame.videoFramePos);
          const isSelected = selectedFramePos === frame.videoFramePos;
          const isCurrent = currentFramePos === frame.videoFramePos;
          const isHovered = hoveredFramePos === frame.videoFramePos;
          return (
            <g
              key={frame.videoFramePos}
              className="cursor-pointer"
              onClick={(event) => {
                event.stopPropagation();
                onSelectFrame(frame.videoFramePos);
              }}
              onMouseEnter={() => onHoverFrame(frame.videoFramePos)}
            >
              <circle
                cx={x}
                cy={122}
                r={isSelected ? 9 : isCurrent ? 7 : 5}
                fill={getFrameTypeColor(frame.frameType)}
                opacity={isHovered || isSelected ? 1 : 0.8}
                stroke={isSelected ? '#18181B' : 'transparent'}
                strokeWidth={isSelected ? 2.5 : 0}
              />
              <line
                x1={x}
                x2={x}
                y1={72}
                y2={112}
                stroke={isCurrent ? '#18181B' : '#D4D4D8'}
                strokeDasharray={isCurrent ? '0' : '3 3'}
                strokeOpacity={isCurrent ? 0.9 : 0.4}
              />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ControlButton({
  children,
  disabled,
  onClick,
}: {
  children: ReactNode;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className="rounded border border-vortex-grey-light/40 dark:border-white/[0.08] px-2.5 py-1 text-[11px] text-vortex-fg-light dark:text-vortex-fg hover:border-vortex-light-blue hover:text-vortex-light-blue disabled:opacity-40 disabled:hover:border-vortex-grey-light/40 disabled:hover:text-inherit transition-colors"
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function compatibilityTrack(track: VideoIndexInfo): VideoTrackInfo {
  return {
    trackId: track.primaryTrackId ?? 1,
    trackLanguage: track.trackLanguage,
    width: track.width,
    height: track.height,
    fpsNum: track.fpsNum,
    fpsDen: track.fpsDen,
    timescale: track.timescale,
    durationTs: track.durationTs,
    durationMs: track.durationMs,
    nalLengthSize: track.nalLengthSize,
    frameCount: track.frameCount,
    gops: track.gops,
    frames: track.frames,
    planningFrames: track.planningFrames,
    samplesByDecode: track.samplesByDecode,
  };
}
