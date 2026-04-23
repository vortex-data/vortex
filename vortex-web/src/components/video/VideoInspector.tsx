// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useMemo, useState } from 'react';
import { DataTable, type CellRenderer } from '../DataTable';
import { DataPreview } from '../explorer/DataPreview';
import type {
  VideoFrameInfo,
  VideoGopInfo,
  VideoIndexInfo,
  VideoTrackInfo,
} from '../swimlane/types';
import {
  describeTrack,
  formatBytes,
  frameClosureRanges,
  getFrameTypeColor,
  gopSampleRanges,
  totalRangeBytes,
  trackSampleRanges,
  type ByteRange,
} from './videoUtils';

type InspectorTab = 'frames' | 'dependencies' | 'fileMap' | 'rawData';

interface VideoInspectorProps {
  videoIndex: VideoIndexInfo;
  selectedTrack: VideoTrackInfo;
  hoveredTrackId: number | null;
  selectedFrame: VideoFrameInfo | null;
  hoveredFrame: VideoFrameInfo | null;
  selectedGop: VideoGopInfo | null;
  hoveredGop: VideoGopInfo | null;
  onSelectTrack: (trackId: number) => void;
  onHoverTrack: (trackId: number | null) => void;
  onSelectFrame: (videoFramePos: number) => void;
  onHoverFrame: (videoFramePos: number | null) => void;
  onHoverGop: (gopPos: number | null) => void;
}

export function VideoInspector({
  videoIndex,
  selectedTrack,
  hoveredTrackId,
  selectedFrame,
  hoveredFrame,
  selectedGop,
  hoveredGop,
  onSelectTrack,
  onHoverTrack,
  onSelectFrame,
  onHoverFrame,
  onHoverGop,
}: VideoInspectorProps) {
  const [activeTab, setActiveTab] = useState<InspectorTab>('fileMap');
  const planningFrameByPos = useMemo(
    () => new Map(selectedTrack.planningFrames.map((frame) => [frame.videoFramePos, frame])),
    [selectedTrack.planningFrames],
  );

  const frameRows = useMemo(
    () =>
      selectedTrack.frames.map((frame) => {
        const planning = planningFrameByPos.get(frame.videoFramePos);
        const sample = selectedTrack.samplesByDecode[frame.globalDecodePos];
        return {
          video_frame_pos: frame.videoFramePos,
          frame_type: frame.frameType,
          gop_pos: frame.gopPos,
          display_pos: frame.displayPos,
          decode_pos: frame.decodePos,
          global_decode_pos: frame.globalDecodePos,
          pts: frame.pts,
          dts: frame.dts,
          duration: frame.duration,
          sample_id: frame.sampleId,
          is_sync: frame.isSync,
          is_reference: frame.isReference,
          dependency_depth: frame.dependencyDepth ?? '',
          ref_l0: frame.refL0GlobalDecodePositions.join(', '),
          ref_l1: frame.refL1GlobalDecodePositions.join(', '),
          sample_byte_offset: sample?.sampleByteOffset ?? frame.sampleByteOffset,
          sample_byte_length: sample?.sampleByteLength ?? frame.sampleByteLength,
          closure_external: planning?.closureExternalDecodePositions.join(', ') ?? '',
        };
      }),
    [planningFrameByPos, selectedTrack.frames, selectedTrack.samplesByDecode],
  );

  const frameTypeRenderer: CellRenderer = (value) => (
    <span
      className="inline-flex rounded-full px-1.5 py-0.5 text-[9px] font-semibold text-white"
      style={{ backgroundColor: getFrameTypeColor(String(value)) }}
    >
      {String(value)}
    </span>
  );

  const frameColumns = [
    'video_frame_pos',
    'frame_type',
    'gop_pos',
    'display_pos',
    'decode_pos',
    'global_decode_pos',
    'pts',
    'dts',
    'duration',
    'sample_id',
    'is_sync',
    'is_reference',
    'dependency_depth',
    'ref_l0',
    'ref_l1',
    'sample_byte_offset',
    'sample_byte_length',
    'closure_external',
  ];

  return (
    <div className="w-[460px] flex-shrink-0 h-full border-l border-vortex-grey-light/60 dark:border-white/[0.08] bg-vortex-white dark:bg-vortex-black flex flex-col min-h-0">
      <div className="border-b border-vortex-grey-light/40 dark:border-white/[0.06] px-3 py-2 space-y-2">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-xs font-medium text-vortex-fg-light dark:text-vortex-fg">
              Semantic Inspector
            </div>
            <div className="text-[10px] text-vortex-grey-dark">
              {describeTrack(selectedTrack)} · GOP {selectedGop?.gopPos ?? '–'} · frame{' '}
              {selectedFrame?.videoFramePos ?? '–'}
            </div>
          </div>
          <div className="text-right text-[10px] text-vortex-grey-dark">
            <div>
              {videoIndex.width}×{videoIndex.height} · {videoIndex.codec}
            </div>
            <div>{videoIndex.tracks.length} tracks indexed</div>
          </div>
        </div>
        <div className="flex gap-1 flex-wrap">
          {([
            ['fileMap', 'File Map'],
            ['frames', 'Frames'],
            ['dependencies', 'Dependencies'],
            ['rawData', 'Raw Data'],
          ] as const).map(([id, label]) => (
            <button
              key={id}
              className={`px-2.5 py-1 text-[10px] font-medium rounded transition-colors ${
                activeTab === id
                  ? 'bg-vortex-light-blue/15 text-vortex-light-blue'
                  : 'text-vortex-grey-dark hover:text-vortex-fg-light dark:hover:text-vortex-fg'
              }`}
              onClick={() => setActiveTab(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-auto">
        {activeTab === 'fileMap' && (
          <VideoFileMap
            videoIndex={videoIndex}
            selectedTrack={selectedTrack}
            hoveredTrackId={hoveredTrackId}
            selectedFrame={selectedFrame}
            hoveredFrame={hoveredFrame}
            selectedGop={selectedGop}
            hoveredGop={hoveredGop}
            onSelectTrack={onSelectTrack}
            onHoverTrack={onHoverTrack}
          />
        )}

        {activeTab === 'frames' && (
          <div className="h-full">
            <DataTable
              columns={frameColumns}
              rows={frameRows}
              onRowClick={(rowIndex) => {
                const row = frameRows[rowIndex];
                if (row) onSelectFrame(Number(row.video_frame_pos));
              }}
              onRowHover={(rowIndex) => {
                if (rowIndex == null) onHoverFrame(null);
                else {
                  const row = frameRows[rowIndex];
                  onHoverFrame(row ? Number(row.video_frame_pos) : null);
                }
              }}
              cellRenderers={{ frame_type: frameTypeRenderer }}
            />
          </div>
        )}

        {activeTab === 'dependencies' && (
          <DependencyGraph
            selectedGop={selectedGop}
            selectedFrame={selectedFrame}
            onSelectFrame={onSelectFrame}
            onHoverFrame={onHoverFrame}
            onHoverGop={onHoverGop}
          />
        )}

        {activeTab === 'rawData' && (
          <div className="h-full min-h-0">
            <div className="px-3 py-2 text-[10px] text-vortex-grey-dark border-b border-vortex-grey-light/20 dark:border-white/[0.04]">
              Raw preview for the currently selected Vortex layout node.
            </div>
            <div className="h-[420px]">
              <DataPreview />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function DependencyGraph({
  selectedGop,
  selectedFrame,
  onSelectFrame,
  onHoverFrame,
  onHoverGop,
}: {
  selectedGop: VideoGopInfo | null;
  selectedFrame: VideoFrameInfo | null;
  onSelectFrame: (videoFramePos: number) => void;
  onHoverFrame: (videoFramePos: number | null) => void;
  onHoverGop: (gopPos: number | null) => void;
}) {
  if (!selectedGop) {
    return <EmptyState message="Select a GOP to inspect its dependency graph." />;
  }

  const width = 980;
  const height = 250;
  const frames = [...selectedGop.frames].sort((left, right) => left.displayPos - right.displayPos);
  const frameByGlobalDecode = new Map(frames.map((frame) => [frame.globalDecodePos, frame]));

  const xForIndex = (index: number) =>
    frames.length <= 1 ? width / 2 : 60 + (index * (width - 120)) / (frames.length - 1);
  const yForFrame = (frame: VideoFrameInfo) => 70 + (frame.dependencyDepth ?? 0) * 34;

  return (
    <div className="p-3 space-y-3" onMouseEnter={() => onHoverGop(selectedGop.gopPos)} onMouseLeave={() => onHoverGop(null)}>
      <div className="text-[10px] text-vortex-grey-dark">
        Display order is left to right. Reference edges only draw for dependencies inside the
        selected GOP; cross-GOP references show as counts on the frame node.
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full rounded border border-vortex-grey-light/30 dark:border-white/[0.06] bg-vortex-grey-lightest/30 dark:bg-white/[0.02]"
      >
        {frames.map((frame, index) => {
          const x = xForIndex(index);
          const y = yForFrame(frame);
          const refs = [...frame.refL0GlobalDecodePositions, ...frame.refL1GlobalDecodePositions];
          const localRefs = Array.from(new Set(refs))
            .map((decodePos) => frameByGlobalDecode.get(decodePos))
            .filter((candidate): candidate is VideoFrameInfo => candidate != null);

          return localRefs.map((refFrame, refIndex) => {
            const refDisplayIndex = frames.findIndex(
              (candidate) => candidate.globalDecodePos === refFrame.globalDecodePos,
            );
            const refX = xForIndex(refDisplayIndex);
            const refY = yForFrame(refFrame);
            const midY = Math.min(y, refY) - 24 - refIndex * 8;
            return (
              <path
                key={`${frame.globalDecodePos}-${refFrame.globalDecodePos}-${refIndex}`}
                d={`M ${x} ${y} Q ${(x + refX) / 2} ${midY} ${refX} ${refY}`}
                fill="none"
                stroke={
                  frame.refL0GlobalDecodePositions.includes(refFrame.globalDecodePos)
                    ? '#2CB9D1'
                    : '#EEB3E1'
                }
                strokeWidth={1.5}
                strokeOpacity={0.75}
              />
            );
          });
        })}

        {frames.map((frame, index) => {
          const x = xForIndex(index);
          const y = yForFrame(frame);
          const refs = [...frame.refL0GlobalDecodePositions, ...frame.refL1GlobalDecodePositions];
          const localRefCount = refs.filter((decodePos) => frameByGlobalDecode.has(decodePos)).length;
          const externalRefCount = refs.length - localRefCount;
          const isSelected = selectedFrame?.videoFramePos === frame.videoFramePos;

          return (
            <g
              key={frame.videoFramePos}
              transform={`translate(${x}, ${y})`}
              onClick={() => onSelectFrame(frame.videoFramePos)}
              onMouseEnter={() => onHoverFrame(frame.videoFramePos)}
              onMouseLeave={() => onHoverFrame(null)}
              className="cursor-pointer"
            >
              <rect
                x={-16}
                y={-14}
                width={32}
                height={28}
                rx={8}
                fill={getFrameTypeColor(frame.frameType)}
                stroke={isSelected ? '#18181B' : 'transparent'}
                strokeWidth={isSelected ? 2.5 : 0}
                opacity={0.95}
              />
              <text
                x={0}
                y={5}
                textAnchor="middle"
                className="fill-white text-[12px] font-semibold"
              >
                {frame.frameType}
              </text>
              <text
                x={0}
                y={-22}
                textAnchor="middle"
                className="fill-vortex-grey-dark text-[10px]"
              >
                f{frame.videoFramePos}
              </text>
              <text
                x={0}
                y={30}
                textAnchor="middle"
                className="fill-vortex-grey-dark text-[9px]"
              >
                d{frame.decodePos}
              </text>
              {externalRefCount > 0 && (
                <g transform="translate(18,-16)">
                  <circle r={8} fill="#18181B" opacity={0.85} />
                  <text x={0} y={3} textAnchor="middle" className="fill-white text-[9px]">
                    {externalRefCount}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function VideoFileMap({
  videoIndex,
  selectedTrack,
  hoveredTrackId,
  selectedFrame,
  hoveredFrame,
  selectedGop,
  hoveredGop,
  onSelectTrack,
  onHoverTrack,
}: {
  videoIndex: VideoIndexInfo;
  selectedTrack: VideoTrackInfo;
  hoveredTrackId: number | null;
  selectedFrame: VideoFrameInfo | null;
  hoveredFrame: VideoFrameInfo | null;
  selectedGop: VideoGopInfo | null;
  hoveredGop: VideoGopInfo | null;
  onSelectTrack: (trackId: number) => void;
  onHoverTrack: (trackId: number | null) => void;
}) {
  const tracks = useMemo(
    () => (videoIndex.tracks.length > 0 ? videoIndex.tracks : [selectedTrack]),
    [selectedTrack, videoIndex.tracks],
  );
  const trackById = useMemo(
    () => new Map(tracks.map((track) => [track.trackId, track])),
    [tracks],
  );
  const hoveredTrack = hoveredTrackId != null ? trackById.get(hoveredTrackId) ?? null : null;

  const trackRanges = useMemo(
    () => new Map(tracks.map((track) => [track.trackId, trackSampleRanges(track)])),
    [tracks],
  );
  const selectedTrackCaches = useMemo(() => {
    const planningFramesByPos = new Map(
      selectedTrack.planningFrames.map((frame) => [frame.videoFramePos, frame]),
    );
    const gopRangesByPos = new Map(
      selectedTrack.gops.map((gop) => [gop.gopPos, gopSampleRanges(selectedTrack, gop)]),
    );
    const frameClosureRangesByPos = new Map(
      selectedTrack.planningFrames.map((frame) => [
        frame.videoFramePos,
        frameClosureRanges(selectedTrack, frame),
      ]),
    );

    return {
      planningFramesByPos,
      gopRangesByPos,
      frameClosureRangesByPos,
    };
  }, [selectedTrack]);

  const focus = useMemo(() => {
    const selectedPlanningFrame =
      selectedFrame != null
        ? selectedTrackCaches.planningFramesByPos.get(selectedFrame.videoFramePos) ?? null
        : null;
    const hoveredPlanningFrame =
      hoveredFrame != null
        ? selectedTrackCaches.planningFramesByPos.get(hoveredFrame.videoFramePos) ?? null
        : null;

    if (hoveredFrame && hoveredPlanningFrame) {
      const ranges =
        selectedTrackCaches.frameClosureRangesByPos.get(hoveredFrame.videoFramePos) ?? [];
      return {
        label: `Frame ${hoveredFrame.videoFramePos} decode closure`,
        track: selectedTrack,
        ranges,
      };
    }
    if (hoveredGop) {
      const ranges = selectedTrackCaches.gopRangesByPos.get(hoveredGop.gopPos) ?? [];
      return {
        label: `Track ${selectedTrack.trackId} GOP ${hoveredGop.gopPos}`,
        track: selectedTrack,
        ranges,
      };
    }
    if (hoveredTrack) {
      return {
        label: `${describeTrack(hoveredTrack)} sample coverage`,
        track: hoveredTrack,
        ranges: trackRanges.get(hoveredTrack.trackId) ?? [],
      };
    }
    if (selectedFrame && selectedPlanningFrame) {
      const ranges =
        selectedTrackCaches.frameClosureRangesByPos.get(selectedFrame.videoFramePos) ?? [];
      return {
        label: `Frame ${selectedFrame.videoFramePos} decode closure`,
        track: selectedTrack,
        ranges,
      };
    }
    if (selectedGop) {
      const ranges = selectedTrackCaches.gopRangesByPos.get(selectedGop.gopPos) ?? [];
      return {
        label: `Track ${selectedTrack.trackId} GOP ${selectedGop.gopPos}`,
        track: selectedTrack,
        ranges,
      };
    }
    return {
      label: `${describeTrack(selectedTrack)} sample coverage`,
      track: selectedTrack,
      ranges: trackRanges.get(selectedTrack.trackId) ?? [],
    };
  }, [
    hoveredFrame,
    hoveredGop,
    hoveredTrack,
    selectedFrame,
    selectedGop,
    selectedTrack,
    selectedTrackCaches,
    trackRanges,
  ]);

  const selectedTrackRanges = trackRanges.get(selectedTrack.trackId) ?? [];
  const showSelectedTrackBase = focus.track.trackId === selectedTrack.trackId;

  return (
    <div className="p-3 space-y-4">
      <div className="grid grid-cols-2 gap-2 text-[10px]">
        <Metric label="Source bytes" value={formatBytes(videoIndex.fileSizeBytes)} />
        <Metric label="Selected track" value={describeTrack(selectedTrack)} />
        <Metric label="Focus" value={focus.label} />
        <Metric
          label="Highlighted bytes"
          value={`${formatBytes(totalRangeBytes(focus.ranges))} across ${focus.ranges.length} ranges`}
        />
      </div>

      <div className="space-y-2">
        <div>
          <div className="text-xs font-medium text-vortex-fg-light dark:text-vortex-fg">
            Track Coverage
          </div>
          <div className="text-[10px] text-vortex-grey-dark">
            Hover a track row to compare which regions of the MP4 each retained frame-rate view
            needs.
          </div>
        </div>

        <div className="space-y-2">
          {tracks.map((track) => {
            const ranges = trackRanges.get(track.trackId) ?? [];
            const isSelected = track.trackId === selectedTrack.trackId;
            const isHovered = track.trackId === hoveredTrackId;
            return (
              <button
                key={track.trackId}
                className={`w-full text-left rounded-lg border px-2.5 py-2 transition-colors ${
                  isSelected
                    ? 'border-vortex-light-blue/80 bg-vortex-light-blue/10'
                    : isHovered
                      ? 'border-vortex-light-blue/50 bg-vortex-grey-lightest/40 dark:bg-white/[0.03]'
                      : 'border-vortex-grey-light/30 dark:border-white/[0.06]'
                }`}
                onClick={() => onSelectTrack(track.trackId)}
                onMouseEnter={() => onHoverTrack(track.trackId)}
                onMouseLeave={() => onHoverTrack(null)}
              >
                <div className="mb-1 flex items-center justify-between gap-3 text-[10px]">
                  <span className="font-medium text-vortex-fg-light dark:text-vortex-fg">
                    {describeTrack(track)}
                  </span>
                  <span className="text-vortex-grey-dark">
                    {ranges.length} merged ranges · {formatBytes(totalRangeBytes(ranges))}
                  </span>
                </div>
                <RangeStrip
                  fileSizeBytes={videoIndex.fileSizeBytes}
                  ranges={ranges}
                  color={isSelected ? '#5971FD' : '#2CB9D1'}
                  opacity={isHovered || isSelected ? 0.9 : 0.45}
                  height={22}
                />
              </button>
            );
          })}
        </div>
      </div>

      <div className="space-y-2">
        <div>
          <div className="text-xs font-medium text-vortex-fg-light dark:text-vortex-fg">
            Focus File Map
          </div>
          <div className="text-[10px] text-vortex-grey-dark">
            Hover a frame, GOP band, or track to highlight the exact byte ranges used for that
            decode target.
          </div>
        </div>

        <div className="rounded-lg border border-vortex-grey-light/30 dark:border-white/[0.06] px-2.5 py-2 space-y-2">
          <div className="flex items-center justify-between gap-3 text-[10px]">
            <span className="font-medium text-vortex-fg-light dark:text-vortex-fg">{focus.label}</span>
            <span className="text-vortex-grey-dark">
              {focus.ranges.length} merged ranges · {formatBytes(totalRangeBytes(focus.ranges))}
            </span>
          </div>

          <RangeStrip
            fileSizeBytes={videoIndex.fileSizeBytes}
            ranges={showSelectedTrackBase ? selectedTrackRanges : []}
            overlayRanges={focus.ranges}
            color="#2CB9D1"
            overlayColor="#5971FD"
            opacity={0.18}
            overlayOpacity={0.92}
            height={34}
          />

          <div className="text-[10px] text-vortex-grey-dark">
            Selected track coverage stays in the background; the brighter overlay follows the
            current focus item.
          </div>
        </div>
      </div>
    </div>
  );
}

function RangeStrip({
  fileSizeBytes,
  ranges,
  overlayRanges,
  color,
  overlayColor,
  opacity = 0.4,
  overlayOpacity = 0.95,
  height,
}: {
  fileSizeBytes: number;
  ranges: ByteRange[];
  overlayRanges?: ByteRange[];
  color: string;
  overlayColor?: string;
  opacity?: number;
  overlayOpacity?: number;
  height: number;
}) {
  return (
    <div
      className="relative overflow-hidden rounded bg-vortex-grey-lightest/70 dark:bg-white/[0.04]"
      style={{ height }}
    >
      {ranges.map((range, index) => (
        <div
          key={`base-${index}-${range.startByteOffset}`}
          className="absolute top-0 bottom-0 rounded-sm"
          style={{
            left: `${(range.startByteOffset / Math.max(fileSizeBytes, 1)) * 100}%`,
            width: `${Math.max((range.byteLength / Math.max(fileSizeBytes, 1)) * 100, 0.25)}%`,
            backgroundColor: color,
            opacity,
          }}
        />
      ))}
      {(overlayRanges ?? []).map((range, index) => (
        <div
          key={`overlay-${index}-${range.startByteOffset}`}
          className="absolute top-0 bottom-0 rounded-sm"
          style={{
            left: `${(range.startByteOffset / Math.max(fileSizeBytes, 1)) * 100}%`,
            width: `${Math.max((range.byteLength / Math.max(fileSizeBytes, 1)) * 100, 0.25)}%`,
            backgroundColor: overlayColor ?? color,
            opacity: overlayOpacity,
          }}
        />
      ))}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-vortex-grey-light/30 dark:border-white/[0.06] px-2 py-1.5">
      <div className="text-vortex-grey-dark">{label}</div>
      <div className="font-mono text-vortex-fg-light dark:text-vortex-fg">{value}</div>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="h-full flex items-center justify-center px-6 text-[11px] text-center text-vortex-grey-dark">
      {message}
    </div>
  );
}
