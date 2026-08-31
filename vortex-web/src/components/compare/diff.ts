// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { ArrayEncodingNode, LayoutTreeNode } from '../swimlane/types';

export type DiffStatus = 'unchanged' | 'changed' | 'added' | 'removed';

export interface TreeDiffNode {
  key: string;
  label: string;
  depth: number;
  status: DiffStatus;
  beforeEncoding?: string;
  afterEncoding?: string;
  beforeMetadataBytes: number;
  afterMetadataBytes: number;
  beforeBufferBytes: number;
  afterBufferBytes: number;
  children: TreeDiffNode[];
}

function layoutChildKey(node: LayoutTreeNode): string {
  const child = node.childType;
  switch (child.kind) {
    case 'root':
      return 'root';
    case 'field':
      return `field:${child.fieldName}`;
    case 'chunk':
      return `chunk:${child.rowOffset}:${node.rowCount}`;
    case 'transparent':
      return `transparent:${child.name}`;
    case 'auxiliary':
      return `auxiliary:${child.name}`;
  }
}

function layoutLabel(node: LayoutTreeNode): string {
  const child = node.childType;
  switch (child.kind) {
    case 'root':
      return 'root';
    case 'field':
      return child.fieldName;
    case 'chunk':
      return `chunk ${child.chunkIndex}`;
    case 'transparent':
    case 'auxiliary':
      return child.name;
  }
}

function arrayBufferBytes(node: ArrayEncodingNode | undefined): number {
  return node?.bufferLengths.reduce((sum, bytes) => sum + bytes, 0) ?? 0;
}

function statusFor(
  beforeEncoding: string | undefined,
  afterEncoding: string | undefined,
  beforeMetadataBytes: number,
  afterMetadataBytes: number,
  beforeBufferBytes: number,
  afterBufferBytes: number,
  children: TreeDiffNode[],
): DiffStatus {
  if (!beforeEncoding) return 'added';
  if (!afterEncoding) return 'removed';
  if (
    beforeEncoding !== afterEncoding ||
    beforeMetadataBytes !== afterMetadataBytes ||
    beforeBufferBytes !== afterBufferBytes ||
    children.some((child) => child.status !== 'unchanged')
  ) {
    return 'changed';
  }
  return 'unchanged';
}

/**
 * Match sibling nodes by a domain key while retaining duplicate keys in source order.
 * This is deliberately not positional: inserting a field must not make every later field
 * appear changed.
 */
function matchChildren<T>(
  before: T[],
  after: T[],
  keyOf: (node: T, index: number) => string,
): Array<{ key: string; before?: T; after?: T }> {
  const afterByKey = new Map<string, T[]>();
  after.forEach((node, index) => {
    const key = keyOf(node, index);
    const matches = afterByKey.get(key) ?? [];
    matches.push(node);
    afterByKey.set(key, matches);
  });

  const pairs: Array<{ key: string; before?: T; after?: T }> = [];
  before.forEach((node, index) => {
    const key = keyOf(node, index);
    const matches = afterByKey.get(key);
    const matched = matches?.shift();
    pairs.push({ key, before: node, after: matched });
    if (matches?.length === 0) afterByKey.delete(key);
  });
  after.forEach((node, index) => {
    const key = keyOf(node, index);
    const matches = afterByKey.get(key);
    if (matches?.[0] === node) {
      matches.shift();
      pairs.push({ key, after: node });
      if (matches.length === 0) afterByKey.delete(key);
    }
  });
  return pairs;
}

function diffArrayNode(
  before: ArrayEncodingNode | undefined,
  after: ArrayEncodingNode | undefined,
  key: string,
  label: string,
  depth: number,
): TreeDiffNode {
  type NamedChild = { key: string; node: ArrayEncodingNode };
  const beforeChildren: NamedChild[] =
    before?.children.map((node, index) => ({
      key: node.name || `child:${index}`,
      node,
    })) ?? [];
  const afterChildren: NamedChild[] =
    after?.children.map((node, index) => ({
      key: node.name || `child:${index}`,
      node,
    })) ?? [];
  const pairs = matchChildren(beforeChildren, afterChildren, (child) => child.key);
  const children = pairs.map((pair, index) => {
    const childLabel = pair.key || `child ${index}`;
    return diffArrayNode(
      pair.before?.node,
      pair.after?.node,
      `${key}/array:${childLabel}`,
      childLabel,
      depth + 1,
    );
  });
  const beforeMetadataBytes = before?.metadataBytes ?? 0;
  const afterMetadataBytes = after?.metadataBytes ?? 0;
  const beforeBufferBytes = arrayBufferBytes(before);
  const afterBufferBytes = arrayBufferBytes(after);
  const beforeEncoding = before?.encoding;
  const afterEncoding = after?.encoding;
  return {
    key,
    label,
    depth,
    status: statusFor(
      beforeEncoding,
      afterEncoding,
      beforeMetadataBytes,
      afterMetadataBytes,
      beforeBufferBytes,
      afterBufferBytes,
      children,
    ),
    beforeEncoding,
    afterEncoding,
    beforeMetadataBytes,
    afterMetadataBytes,
    beforeBufferBytes,
    afterBufferBytes,
    children,
  };
}

function diffLayoutNode(
  before: LayoutTreeNode | undefined,
  after: LayoutTreeNode | undefined,
  key: string,
  depth: number,
): TreeDiffNode {
  // Expanded array nodes are a UI projection of `arrayEncodingTree`, not layout children.
  // Excluding them prevents a file explored before comparison from appearing structurally
  // different from an otherwise identical freshly opened file.
  const beforeChildren = before?.children.filter((node) => !node.isArrayNode) ?? [];
  const afterChildren = after?.children.filter((node) => !node.isArrayNode) ?? [];
  const pairs = matchChildren(beforeChildren, afterChildren, (node) => layoutChildKey(node));
  const children = pairs.map((pair) =>
    diffLayoutNode(pair.before, pair.after, `${key}/${pair.key}`, depth + 1),
  );

  const beforeArray = before?.arrayEncodingTree;
  const afterArray = after?.arrayEncodingTree;
  if (beforeArray || afterArray) {
    children.push(diffArrayNode(beforeArray, afterArray, `${key}/array`, 'array', depth + 1));
  }

  const beforeMetadataBytes = before?.metadataBytes ?? 0;
  const afterMetadataBytes = after?.metadataBytes ?? 0;
  const beforeEncoding = before?.encoding;
  const afterEncoding = after?.encoding;
  return {
    key,
    label: after ? layoutLabel(after) : before ? layoutLabel(before) : key,
    depth,
    status: statusFor(
      beforeEncoding,
      afterEncoding,
      beforeMetadataBytes,
      afterMetadataBytes,
      0,
      0,
      children,
    ),
    beforeEncoding,
    afterEncoding,
    beforeMetadataBytes,
    afterMetadataBytes,
    beforeBufferBytes: 0,
    afterBufferBytes: 0,
    children,
  };
}

export function diffLayoutTrees(before: LayoutTreeNode, after: LayoutTreeNode): TreeDiffNode {
  return diffLayoutNode(before, after, 'root', 0);
}

export function flattenDiff(root: TreeDiffNode, changesOnly: boolean): TreeDiffNode[] {
  const rows: TreeDiffNode[] = [];
  function visit(node: TreeDiffNode) {
    if (!changesOnly || node.status !== 'unchanged') rows.push(node);
    node.children.forEach(visit);
  }
  visit(root);
  return rows;
}
