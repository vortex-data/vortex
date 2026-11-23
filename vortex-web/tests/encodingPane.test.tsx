// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterEach, describe, expect, test } from 'bun:test';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { EncodingPane } from '../src/components/detail/EncodingPane';
import type { ArrayEncodingNode, LayoutTreeNode } from '../src/components/swimlane/types';
import type { VortexFileContextValue } from '../src/contexts/VortexFileContext';
import { VortexFileContext } from '../src/contexts/VortexFileContextCore';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (err: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (err: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function encodingTree(encoding: string): ArrayEncodingNode {
  return {
    encoding,
    dtype: 'u8',
    metadataBytes: 0,
    numBuffers: 0,
    bufferLengths: [],
    bufferNames: [],
    children: [],
    childNames: [],
  };
}

function makeNode(id: string, extra: Partial<LayoutTreeNode> = {}): LayoutTreeNode {
  return {
    id,
    encoding: 'vortex.flat',
    dtype: 'u8',
    rowCount: 1,
    rowOffset: 0,
    metadataBytes: 0,
    segmentIds: [1],
    childType: { kind: 'root' },
    children: [],
    ...extra,
  };
}

let mountedRoot: Root | null = null;
let mountedContainer: HTMLDivElement | null = null;

function createHarness(fetchEncodingTree: VortexFileContextValue['fetchEncodingTree']) {
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  mountedRoot = root;
  mountedContainer = container;
  const ctx = { fetchEncodingTree } as VortexFileContextValue;
  return {
    container,
    render: (node: LayoutTreeNode) =>
      act(async () => {
        root.render(
          <VortexFileContext.Provider value={ctx}>
            <EncodingPane node={node} />
          </VortexFileContext.Provider>,
        );
      }),
  };
}

afterEach(async () => {
  if (mountedRoot) {
    const root = mountedRoot;
    mountedRoot = null;
    await act(async () => {
      root.unmount();
    });
  }
  mountedContainer?.remove();
  mountedContainer = null;
});

describe('EncodingPane', () => {
  test('shows inline encoding immediately when switching away from a pending fetch', async () => {
    const req = deferred<ArrayEncodingNode | null>();
    const { container, render } = createHarness(() => req.promise);

    const nodeA = makeNode('A');
    const nodeB = makeNode('B', { arrayEncodingTree: encodingTree('vortex.constant') });

    await render(nodeA);
    expect(container.textContent).toContain('Reading the encoding tree…');

    await render(nodeB);
    expect(container.textContent).toContain('constant');
    expect(container.textContent).not.toContain('Reading the encoding tree…');
  });

  test('clears a previously fetched tree when switching to an empty node', async () => {
    const req = deferred<ArrayEncodingNode | null>();
    const { container, render } = createHarness(() => req.promise);

    const nodeA = makeNode('A');
    const nodeC = makeNode('C', { segmentIds: [] });

    await render(nodeA);
    await act(async () => {
      req.resolve(encodingTree('vortex.primitive'));
    });
    expect(container.textContent).toContain('primitive');

    await render(nodeC);
    expect(container.textContent).toContain('This node has no array encoding tree.');
    expect(container.textContent).not.toContain('primitive');
  });

  // Component contract test: current DetailPanel passes a selection snapshot, so same-id inline
  // updates do not happen in the app today, but EncodingPane still clears loading if props update.
  test('clears loading when inline metadata is added to a pending node', async () => {
    const req = deferred<ArrayEncodingNode | null>();
    const { container, render } = createHarness(() => req.promise);

    await render(makeNode('A'));
    expect(container.textContent).toContain('Reading the encoding tree…');

    await render(makeNode('A', { arrayEncodingTree: encodingTree('vortex.bitpacked') }));
    expect(container.textContent).toContain('bitpacked');
    expect(container.textContent).not.toContain('Reading the encoding tree…');
  });
});
