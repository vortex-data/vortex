// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterEach, describe, expect, test } from 'bun:test';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { BuffersPane } from '../src/components/detail/BuffersPane';
import type { LayoutTreeNode } from '../src/components/swimlane/types';
import {
  VortexFileContext,
  type VortexFileContextValue,
} from '../src/contexts/VortexFileContextCore';

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

const twoBufferNode: LayoutTreeNode = {
  id: 'layout-0/arr',
  name: 'arr',
  encoding: 'vortex.primitive',
  dtype: 'u8',
  byteSize: 2,
  rowCount: 1,
  metadataBytes: 0,
  bufferLengths: [1, 1],
  bufferNames: ['first', 'second'],
  children: [],
};

let mountedRoot: Root | null = null;
let mountedContainer: HTMLDivElement | null = null;

async function mountPane(
  fetchArrayBuffer: VortexFileContextValue['fetchArrayBuffer'],
): Promise<HTMLDivElement> {
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  mountedRoot = root;
  mountedContainer = container;

  const ctx = { fetchArrayBuffer } as VortexFileContextValue;
  await act(async () => {
    root.render(
      <VortexFileContext.Provider value={ctx}>
        <BuffersPane node={twoBufferNode} />
      </VortexFileContext.Provider>,
    );
  });
  return container;
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

describe('BuffersPane', () => {
  test('ignores a stale resolution when another buffer is selected on the same node', async () => {
    const reqs = new Map<number, Deferred<Uint8Array>>();
    const container = await mountPane((_layoutId, _path, bufferIndex) => {
      const d = deferred<Uint8Array>();
      reqs.set(bufferIndex, d);
      return d.promise;
    });

    const buttons = container.querySelectorAll('button');
    expect(buttons).toHaveLength(2);

    await act(async () => {
      buttons[1]?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    await act(async () => {
      reqs.get(1)?.resolve(new Uint8Array([0xbb]));
    });
    expect(container.textContent).toContain('bb');
    expect(container.textContent).not.toContain('Reading the buffer…');

    await act(async () => {
      reqs.get(0)?.resolve(new Uint8Array([0xaa]));
    });
    expect(container.textContent).toContain('bb');
    expect(container.textContent).not.toContain('aa');
  });

  test('ignores a stale rejection while a newer buffer read on the same node is still loading', async () => {
    const reqs = new Map<number, Deferred<Uint8Array>>();
    const container = await mountPane((_layoutId, _path, bufferIndex) => {
      const d = deferred<Uint8Array>();
      reqs.set(bufferIndex, d);
      return d.promise;
    });

    const buttons = container.querySelectorAll('button');
    await act(async () => {
      buttons[1]?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(container.textContent).toContain('Reading the buffer…');

    await act(async () => {
      reqs.get(0)?.reject(new Error('stale failure'));
    });
    expect(container.textContent).toContain('Reading the buffer…');
    expect(container.textContent).not.toContain('stale failure');

    await act(async () => {
      reqs.get(1)?.resolve(new Uint8Array([0x22]));
    });
    expect(container.textContent).toContain('22');
    expect(container.textContent).not.toContain('Reading the buffer…');
  });
});
