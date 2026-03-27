// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { LayoutTreeNode, LayoutChildKind } from '../components/swimlane/types';

let nextSegmentId = 0;

export function resetSegmentIds() {
  nextSegmentId = 0;
}

function allocSegments(count: number): number[] {
  const ids = [];
  for (let i = 0; i < count; i++) {
    ids.push(nextSegmentId++);
  }
  return ids;
}

export function makeFlat(opts: {
  id: string;
  childType: LayoutChildKind;
  dtype: string;
  rowCount: number;
  rowOffset: number;
  metadataBytes?: number;
}): LayoutTreeNode {
  return {
    id: opts.id,
    encoding: 'vortex.flat',
    dtype: opts.dtype,
    rowCount: opts.rowCount,
    rowOffset: opts.rowOffset,
    metadataBytes: opts.metadataBytes ?? 64,
    segmentIds: allocSegments(1),
    childType: opts.childType,
    children: [],
  };
}

export function makeChunked(opts: {
  id: string;
  childType: LayoutChildKind;
  dtype: string;
  rowCount: number;
  rowOffset: number;
  chunkCount: number;
  childEncoding?: string;
  metadataBytes?: number;
}): LayoutTreeNode {
  const rowsPerChunk = Math.ceil(opts.rowCount / opts.chunkCount);
  const children: LayoutTreeNode[] = [];

  for (let i = 0; i < opts.chunkCount; i++) {
    const chunkOffset = opts.rowOffset + i * rowsPerChunk;
    const chunkRows = Math.min(rowsPerChunk, opts.rowOffset + opts.rowCount - chunkOffset);
    if (chunkRows <= 0) break;

    children.push(
      makeFlat({
        id: `${opts.id}.[${i}]`,
        childType: { kind: 'chunk', chunkIndex: i, rowOffset: chunkOffset },
        dtype: opts.dtype,
        rowCount: chunkRows,
        rowOffset: chunkOffset,
        metadataBytes: 32,
      }),
    );
  }

  return {
    id: opts.id,
    encoding: 'vortex.chunked',
    dtype: opts.dtype,
    rowCount: opts.rowCount,
    rowOffset: opts.rowOffset,
    metadataBytes: opts.metadataBytes ?? 128,
    segmentIds: allocSegments(1),
    childType: opts.childType,
    children,
  };
}

export function makeStruct(opts: {
  id: string;
  childType: LayoutChildKind;
  dtype: string;
  rowCount: number;
  rowOffset: number;
  fields: LayoutTreeNode[];
  metadataBytes?: number;
}): LayoutTreeNode {
  return {
    id: opts.id,
    encoding: 'vortex.struct',
    dtype: opts.dtype,
    rowCount: opts.rowCount,
    rowOffset: opts.rowOffset,
    metadataBytes: opts.metadataBytes ?? 96,
    segmentIds: allocSegments(1),
    childType: opts.childType,
    children: opts.fields,
  };
}

export function makeDict(opts: {
  id: string;
  childType: LayoutChildKind;
  dtype: string;
  rowCount: number;
  rowOffset: number;
  codesDtype?: string;
  metadataBytes?: number;
}): LayoutTreeNode {
  return {
    id: opts.id,
    encoding: 'vortex.dict',
    dtype: opts.dtype,
    rowCount: opts.rowCount,
    rowOffset: opts.rowOffset,
    metadataBytes: opts.metadataBytes ?? 64,
    segmentIds: allocSegments(1),
    childType: opts.childType,
    children: [
      makeFlat({
        id: `${opts.id}.codes`,
        childType: { kind: 'transparent', name: 'codes' },
        dtype: opts.codesDtype ?? 'u16',
        rowCount: opts.rowCount,
        rowOffset: opts.rowOffset,
      }),
      makeFlat({
        id: `${opts.id}.values`,
        childType: { kind: 'transparent', name: 'values' },
        dtype: opts.dtype,
        rowCount: opts.rowCount,
        rowOffset: opts.rowOffset,
      }),
    ],
  };
}

export function makeZoned(opts: {
  id: string;
  childType: LayoutChildKind;
  dtype: string;
  rowCount: number;
  rowOffset: number;
  zoneCount: number;
  metadataBytes?: number;
}): LayoutTreeNode {
  const rowsPerZone = Math.ceil(opts.rowCount / opts.zoneCount);
  const children: LayoutTreeNode[] = [];

  // Zones auxiliary child
  const zoneChildren: LayoutTreeNode[] = [];
  for (let i = 0; i < opts.zoneCount; i++) {
    const zoneOffset = opts.rowOffset + i * rowsPerZone;
    const zoneRows = Math.min(rowsPerZone, opts.rowOffset + opts.rowCount - zoneOffset);
    if (zoneRows <= 0) break;

    zoneChildren.push(
      makeFlat({
        id: `${opts.id}.zones.[${i}]`,
        childType: { kind: 'chunk', chunkIndex: i, rowOffset: zoneOffset },
        dtype: opts.dtype,
        rowCount: zoneRows,
        rowOffset: zoneOffset,
      }),
    );
  }

  children.push({
    id: `${opts.id}.data`,
    encoding: 'vortex.chunked',
    dtype: opts.dtype,
    rowCount: opts.rowCount,
    rowOffset: opts.rowOffset,
    metadataBytes: 64,
    segmentIds: allocSegments(1),
    childType: { kind: 'transparent', name: 'data' },
    children: zoneChildren,
  });

  // Zone map stats (auxiliary)
  children.push(
    makeFlat({
      id: `${opts.id}.zone_map`,
      childType: { kind: 'auxiliary', name: 'zone_map' },
      dtype: `{min=${opts.dtype}, max=${opts.dtype}}`,
      rowCount: opts.zoneCount,
      rowOffset: 0,
    }),
  );

  return {
    id: opts.id,
    encoding: 'vortex.zonemap',
    dtype: opts.dtype,
    rowCount: opts.rowCount,
    rowOffset: opts.rowOffset,
    metadataBytes: opts.metadataBytes ?? 256,
    segmentIds: allocSegments(1),
    childType: opts.childType,
    children,
  };
}
