// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { LayoutTreeNode, SegmentMapEntry } from '../components/swimlane/types';
import {
  resetSegmentIds,
  makeFlat,
  makeChunked,
  makeStruct,
  makeDict,
  makeZoned,
} from './generators';

/**
 * Simple: single flat column
 */
export function simpleMock(): LayoutTreeNode {
  resetSegmentIds();
  return makeFlat({
    id: 'root',
    childType: { kind: 'root' },
    dtype: 'i64',
    rowCount: 10000,
    rowOffset: 0,
    metadataBytes: 64,
  });
}

/**
 * Orders: 7 columns, mixed encodings — the canonical demo dataset
 */
export function ordersMock(): LayoutTreeNode {
  resetSegmentIds();
  const totalRows = 100_000;

  return makeStruct({
    id: 'root',
    childType: { kind: 'root' },
    dtype:
      '{order_id=i64, is_active=bool, customer={id=i64, name=utf8}, items=list<struct>, amount=f64, metadata=struct, status=utf8}',
    rowCount: totalRows,
    rowOffset: 0,
    fields: [
      makeFlat({
        id: 'root.order_id',
        childType: { kind: 'field', fieldName: 'order_id' },
        dtype: 'i64',
        rowCount: totalRows,
        rowOffset: 0,
        metadataBytes: 64,
      }),
      makeFlat({
        id: 'root.is_active',
        childType: { kind: 'field', fieldName: 'is_active' },
        dtype: 'bool',
        rowCount: totalRows,
        rowOffset: 0,
        metadataBytes: 32,
      }),
      makeStruct({
        id: 'root.customer',
        childType: { kind: 'field', fieldName: 'customer' },
        dtype: '{id=i64, name=utf8}',
        rowCount: totalRows,
        rowOffset: 0,
        fields: [
          makeChunked({
            id: 'root.customer.id',
            childType: { kind: 'field', fieldName: 'id' },
            dtype: 'i64',
            rowCount: totalRows,
            rowOffset: 0,
            chunkCount: 50,
          }),
          makeChunked({
            id: 'root.customer.name',
            childType: { kind: 'field', fieldName: 'name' },
            dtype: 'utf8',
            rowCount: totalRows,
            rowOffset: 0,
            chunkCount: 8,
          }),
        ],
      }),
      makeFlat({
        id: 'root.items',
        childType: { kind: 'field', fieldName: 'items' },
        dtype: 'list<struct>',
        rowCount: totalRows,
        rowOffset: 0,
        metadataBytes: 128,
      }),
      makeZoned({
        id: 'root.amount',
        childType: { kind: 'field', fieldName: 'amount' },
        dtype: 'f64',
        rowCount: totalRows,
        rowOffset: 0,
        zoneCount: 5,
      }),
      makeFlat({
        id: 'root.metadata',
        childType: { kind: 'field', fieldName: 'metadata' },
        dtype: 'struct',
        rowCount: totalRows,
        rowOffset: 0,
        metadataBytes: 256,
      }),
      makeDict({
        id: 'root.status',
        childType: { kind: 'field', fieldName: 'status' },
        dtype: 'utf8',
        rowCount: totalRows,
        rowOffset: 0,
      }),
    ],
  });
}

/**
 * Wide: 200 columns (to test search and scrolling)
 */
export function wideMock(): LayoutTreeNode {
  resetSegmentIds();
  const totalRows = 50_000;
  const dtypes = ['i32', 'i64', 'f32', 'f64', 'utf8', 'bool', 'u16', 'u32'];

  const fields: LayoutTreeNode[] = Array.from({ length: 200 }, (_, i) => {
    const dtype = dtypes[i % dtypes.length];
    return makeFlat({
      id: `root.col_${i}`,
      childType: { kind: 'field', fieldName: `col_${i}` },
      dtype,
      rowCount: totalRows,
      rowOffset: 0,
    });
  });

  return makeStruct({
    id: 'root',
    childType: { kind: 'root' },
    dtype: '{...200 columns}',
    rowCount: totalRows,
    rowOffset: 0,
    fields,
  });
}

/**
 * Deep: nested structs (3 levels deep)
 */
export function deepMock(): LayoutTreeNode {
  resetSegmentIds();
  const totalRows = 25_000;

  return makeStruct({
    id: 'root',
    childType: { kind: 'root' },
    dtype:
      '{user={profile={address={street=utf8, city=utf8, zip=u32}, name=utf8, age=u8}, orders=list<struct>}}',
    rowCount: totalRows,
    rowOffset: 0,
    fields: [
      makeStruct({
        id: 'root.user',
        childType: { kind: 'field', fieldName: 'user' },
        dtype: '{profile={...}, orders=list<struct>}',
        rowCount: totalRows,
        rowOffset: 0,
        fields: [
          makeStruct({
            id: 'root.user.profile',
            childType: { kind: 'field', fieldName: 'profile' },
            dtype: '{address={...}, name=utf8, age=u8}',
            rowCount: totalRows,
            rowOffset: 0,
            fields: [
              makeStruct({
                id: 'root.user.profile.address',
                childType: { kind: 'field', fieldName: 'address' },
                dtype: '{street=utf8, city=utf8, zip=u32}',
                rowCount: totalRows,
                rowOffset: 0,
                fields: [
                  makeFlat({
                    id: 'root.user.profile.address.street',
                    childType: { kind: 'field', fieldName: 'street' },
                    dtype: 'utf8',
                    rowCount: totalRows,
                    rowOffset: 0,
                  }),
                  makeFlat({
                    id: 'root.user.profile.address.city',
                    childType: { kind: 'field', fieldName: 'city' },
                    dtype: 'utf8',
                    rowCount: totalRows,
                    rowOffset: 0,
                  }),
                  makeFlat({
                    id: 'root.user.profile.address.zip',
                    childType: { kind: 'field', fieldName: 'zip' },
                    dtype: 'u32',
                    rowCount: totalRows,
                    rowOffset: 0,
                  }),
                ],
              }),
              makeFlat({
                id: 'root.user.profile.name',
                childType: { kind: 'field', fieldName: 'name' },
                dtype: 'utf8',
                rowCount: totalRows,
                rowOffset: 0,
              }),
              makeFlat({
                id: 'root.user.profile.age',
                childType: { kind: 'field', fieldName: 'age' },
                dtype: 'u8',
                rowCount: totalRows,
                rowOffset: 0,
              }),
            ],
          }),
          makeChunked({
            id: 'root.user.orders',
            childType: { kind: 'field', fieldName: 'orders' },
            dtype: 'list<struct>',
            rowCount: totalRows,
            rowOffset: 0,
            chunkCount: 5,
          }),
        ],
      }),
    ],
  });
}

/**
 * Heavy chunks: a single column with 500 chunks (to test grouping)
 */
export function heavyChunksMock(): LayoutTreeNode {
  resetSegmentIds();
  const totalRows = 1_000_000;

  return makeStruct({
    id: 'root',
    childType: { kind: 'root' },
    dtype: '{values=i64}',
    rowCount: totalRows,
    rowOffset: 0,
    fields: [
      makeChunked({
        id: 'root.values',
        childType: { kind: 'field', fieldName: 'values' },
        dtype: 'i64',
        rowCount: totalRows,
        rowOffset: 0,
        chunkCount: 500,
      }),
    ],
  });
}

/**
 * Gapped: two columns whose chunks are interleaved on disk (temp[0], count[0],
 * temp[1], count[1], …). Plotting by physical byte offset shows each column's
 * storage split by the other's — a gap in the column representation. Returns a
 * hand-built segment map so the interleaving is explicit.
 */
export function gappedMock(): {
  layout: LayoutTreeNode;
  segments: SegmentMapEntry[];
  totalBytes: number;
} {
  const rowsPerChunk = 30_000;
  const totalRows = rowsPerChunk * 3;
  const chunkBytes = 100_000;

  const chunk = (columnId: string, dtype: string, i: number, segId: number): LayoutTreeNode => ({
    id: `${columnId}.[${i}]`,
    encoding: 'vortex.flat',
    dtype,
    rowCount: rowsPerChunk,
    rowOffset: i * rowsPerChunk,
    metadataBytes: 32,
    segmentIds: [segId],
    childType: { kind: 'chunk', chunkIndex: i, rowOffset: i * rowsPerChunk },
    children: [],
  });

  const column = (
    columnId: string,
    fieldName: string,
    dtype: string,
    segIds: number[],
  ): LayoutTreeNode => ({
    id: columnId,
    encoding: 'vortex.chunked',
    dtype,
    rowCount: totalRows,
    rowOffset: 0,
    metadataBytes: 64,
    segmentIds: [],
    childType: { kind: 'field', fieldName },
    children: segIds.map((segId, i) => chunk(columnId, dtype, i, segId)),
  });

  const layout: LayoutTreeNode = {
    id: 'root',
    encoding: 'vortex.struct',
    dtype: '{temp=f64, count=i64}',
    rowCount: totalRows,
    rowOffset: 0,
    metadataBytes: 96,
    segmentIds: [],
    childType: { kind: 'root' },
    children: [
      column('root.temp', 'temp', 'f64', [0, 2, 4]),
      column('root.count', 'count', 'i64', [1, 3, 5]),
    ],
  };

  // Segment i sits at byte offset i * chunkBytes, so even segments (temp) and odd
  // segments (count) alternate on disk — each column is physically fragmented.
  const segments: SegmentMapEntry[] = [0, 1, 2, 3, 4, 5].map((index) => ({
    index,
    byteOffset: index * chunkBytes,
    byteLength: chunkBytes,
    alignment: 64,
    column: index % 2 === 0 ? 'temp' : 'count',
    layoutPath: index % 2 === 0 ? `root.temp.[${index / 2}]` : `root.count.[${(index - 1) / 2}]`,
  }));

  return { layout, segments, totalBytes: 6 * chunkBytes };
}
