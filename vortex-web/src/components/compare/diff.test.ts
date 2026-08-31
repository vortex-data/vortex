// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import type { ArrayEncodingNode, LayoutChildKind, LayoutTreeNode } from '../swimlane/types';
import { diffLayoutTrees, flattenDiff } from './diff';

function node({
  id,
  childType,
  children = [],
  encoding = 'vortex.flat',
  metadataBytes = 0,
  isArrayNode,
}: {
  id: string;
  childType: LayoutChildKind;
  children?: LayoutTreeNode[];
  encoding?: string;
  metadataBytes?: number;
  isArrayNode?: boolean;
}): LayoutTreeNode {
  return {
    id,
    encoding,
    dtype: 'utf8',
    rowCount: 100,
    rowOffset: 0,
    metadataBytes,
    segmentIds: [],
    childType,
    children,
    isArrayNode,
  };
}

function field(name: string, options: Partial<Parameters<typeof node>[0]> = {}): LayoutTreeNode {
  return node({
    id: `root.${name}`,
    childType: { kind: 'field', fieldName: name },
    ...options,
  });
}

function root(children: LayoutTreeNode[]): LayoutTreeNode {
  return node({ id: 'root', childType: { kind: 'root' }, children, encoding: 'vortex.struct' });
}

function arrayNode(
  name: string,
  encoding = 'vortex.primitive',
  children: ArrayEncodingNode[] = [],
): ArrayEncodingNode {
  return {
    name,
    encoding,
    dtype: 'i32',
    metadataBytes: 0,
    numBuffers: 0,
    bufferLengths: [],
    bufferNames: [],
    children,
  };
}

describe('semantic layout diff', () => {
  it('matches fields by name when a field is inserted', () => {
    const result = diffLayoutTrees(
      root([field('a'), field('b')]),
      root([field('a'), field('x'), field('b')]),
    );

    assert.deepEqual(
      result.children.map(({ label, status }) => [label, status]),
      [
        ['a', 'unchanged'],
        ['b', 'unchanged'],
        ['x', 'added'],
      ],
    );
    assert.deepEqual(
      flattenDiff(result, true).map(({ label, status }) => [label, status]),
      [
        ['root', 'changed'],
        ['x', 'added'],
      ],
    );
  });

  it('reports encoding and metadata changes on the matching field', () => {
    const before = root([field('value', { encoding: 'vortex.dict', metadataBytes: 12 })]);
    const after = root([field('value', { encoding: 'vortex.on_pair', metadataBytes: 20 })]);
    const changed = diffLayoutTrees(before, after).children[0];

    assert.deepEqual(
      {
        label: changed?.label,
        status: changed?.status,
        beforeEncoding: changed?.beforeEncoding,
        afterEncoding: changed?.afterEncoding,
        beforeMetadataBytes: changed?.beforeMetadataBytes,
        afterMetadataBytes: changed?.afterMetadataBytes,
      },
      {
        label: 'value',
        status: 'changed',
        beforeEncoding: 'vortex.dict',
        afterEncoding: 'vortex.on_pair',
        beforeMetadataBytes: 12,
        afterMetadataBytes: 20,
      },
    );
  });

  it('ignores expanded array nodes that exist only in the rendered tree', () => {
    const expandedArrayNode = field('array child', { isArrayNode: true });
    const before = root([field('value')]);
    const after = root([field('value'), expandedArrayNode]);

    assert.equal(diffLayoutTrees(before, after).status, 'unchanged');
  });

  it('matches streamed array children by the name carried on each node', () => {
    const before = field('value');
    before.arrayEncodingTree = arrayNode('array', 'vortex.struct', [
      arrayNode('a'),
      arrayNode('b'),
    ]);
    const after = field('value');
    after.arrayEncodingTree = arrayNode('array', 'vortex.struct', [
      arrayNode('a'),
      arrayNode('inserted', 'vortex.constant'),
      arrayNode('b'),
    ]);

    const arrayDiff = diffLayoutTrees(root([before]), root([after])).children[0]?.children[0];
    assert.deepEqual(
      arrayDiff?.children.map(({ label, status }) => [label, status]),
      [
        ['a', 'unchanged'],
        ['b', 'unchanged'],
        ['inserted', 'added'],
      ],
    );
  });
});
