// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterEach, describe, expect, test } from 'bun:test';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { TreePanel } from '../src/components/explorer/TreePanel';
import { SwimlaneOverview } from '../src/components/swimlane/SwimlaneOverview';
import type { LayoutTreeNode } from '../src/components/swimlane/types';
import { SelectionContext, type SelectionContextValue } from '../src/contexts/SelectionContextCore';
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

function makeFlatFile(expandArrayTree: (nodeId: string) => Promise<void>): VortexFileContextValue {
  const layoutTree: LayoutTreeNode = {
    id: 'root',
    encoding: 'vortex.flat',
    dtype: 'u8',
    childType: { kind: 'root' },
    rowCount: 1,
    rowOffset: 0,
    metadataBytes: 0,
    segmentIds: [0],
    children: [],
  };
  return {
    fileName: 'test.vortex',
    fileSize: 5,
    rowCount: 1,
    version: 1,
    dtype: 'u8',
    layoutTree,
    segments: [{ index: 0, byteLength: 5, byteOffset: 0, alignment: 1 }],
    fileStructure: { fileSize: 5, version: 1, segments: [], footerSize: 0 },
    expandArrayTree,
  } as unknown as VortexFileContextValue;
}

const emptySelection: SelectionContextValue = {
  state: { selectedNode: null, selectedNodeId: null, hoveredNodeId: null },
  selectNode() {},
  hoverNode() {},
};

let mountedRoot: Root | null = null;
let mountedContainer: HTMLDivElement | null = null;

async function mountView(
  element: React.ReactNode,
  file: VortexFileContextValue,
): Promise<HTMLDivElement> {
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  mountedRoot = root;
  mountedContainer = container;
  await act(async () => {
    root.render(
      <VortexFileContext.Provider value={file}>
        <SelectionContext.Provider value={emptySelection}>{element}</SelectionContext.Provider>
      </VortexFileContext.Provider>,
    );
  });
  return container;
}

async function flushMicrotasks() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
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

for (const [name, renderView, toggleSelector] of [
  ['TreePanel', () => <TreePanel />, '.vx-tree-toggle'],
  ['SwimlaneOverview', () => <SwimlaneOverview />, '.x-lane-toggle'],
] as const) {
  describe(`${name} array tree expansion`, () => {
    test('retries after a failed read when collapsed and expanded again without unhandled rejections', async () => {
      let calls = 0;
      let unhandled = 0;
      const onUnhandled = () => {
        unhandled++;
      };
      process.on('unhandledRejection', onUnhandled);
      try {
        const file = makeFlatFile(() => {
          calls++;
          return calls === 1 ? Promise.reject(new Error('temporary')) : Promise.resolve();
        });

        const container = await mountView(renderView(), file);
        await flushMicrotasks();

        // Initial expansion failed once and must not auto-retry in a loop.
        expect(calls).toBe(1);
        expect(unhandled).toBe(0);

        const toggle = () =>
          act(async () => {
            const btn = container.querySelector(toggleSelector);
            expect(btn).not.toBeNull();
            btn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
          });

        // Collapse, then expand again: should issue exactly one retry.
        await toggle();
        await toggle();
        await flushMicrotasks();

        expect(calls).toBe(2);
        expect(unhandled).toBe(0);
      } finally {
        process.off('unhandledRejection', onUnhandled);
      }
    });

    test('shares an in-flight request across collapse and re-expand while pending', async () => {
      let calls = 0;
      const pending = deferred<void>();
      const file = makeFlatFile(() => {
        calls++;
        return pending.promise;
      });

      const container = await mountView(renderView(), file);
      expect(calls).toBe(1);

      const toggle = () =>
        act(async () => {
          const btn = container.querySelector(toggleSelector);
          expect(btn).not.toBeNull();
          btn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        });

      await toggle();
      await toggle();
      expect(calls).toBe(1);

      await act(async () => {
        pending.resolve(undefined);
      });
      expect(calls).toBe(1);
    });

    test('does not auto-retry a persistently failing node when another node toggles', async () => {
      const calls: string[] = [];
      const colA: LayoutTreeNode = {
        id: 'root.a',
        encoding: 'vortex.flat',
        dtype: 'u8',
        childType: { kind: 'field', fieldName: 'a' },
        rowCount: 1,
        rowOffset: 0,
        metadataBytes: 0,
        segmentIds: [0],
        children: [],
      };
      const colB: LayoutTreeNode = {
        id: 'root.b',
        encoding: 'vortex.flat',
        dtype: 'u8',
        childType: { kind: 'field', fieldName: 'b' },
        rowCount: 1,
        rowOffset: 0,
        metadataBytes: 0,
        segmentIds: [1],
        children: [],
      };
      const file = {
        fileName: 'test.vortex',
        fileSize: 10,
        rowCount: 1,
        version: 1,
        dtype: '{a=u8, b=u8}',
        layoutTree: {
          id: 'root',
          encoding: 'vortex.struct',
          dtype: '{a=u8, b=u8}',
          childType: { kind: 'root' },
          rowCount: 1,
          rowOffset: 0,
          metadataBytes: 0,
          segmentIds: [],
          children: [colA, colB],
        },
        segments: [
          { index: 0, byteLength: 5, byteOffset: 0, alignment: 1 },
          { index: 1, byteLength: 5, byteOffset: 5, alignment: 1 },
        ],
        fileStructure: { fileSize: 10, version: 1, segments: [], footerSize: 0 },
        expandArrayTree: (nodeId: string) => {
          calls.push(nodeId);
          return nodeId === 'root.a' ? Promise.reject(new Error('persistent')) : Promise.resolve();
        },
      } as unknown as VortexFileContextValue;

      const container = await mountView(renderView(), file);
      const toggles = Array.from(container.querySelectorAll(toggleSelector));
      // Index 0 is root, 1 is colA, 2 is colB.
      expect(toggles.length).toBe(3);

      await act(async () => {
        toggles[1].dispatchEvent(new MouseEvent('click', { bubbles: true }));
      });
      await flushMicrotasks();
      expect(calls).toEqual(['root.a']);

      // Expanding colB changes `expanded` while colA stays open: colA must not retry automatically.
      await act(async () => {
        toggles[2].dispatchEvent(new MouseEvent('click', { bubbles: true }));
      });
      await flushMicrotasks();
      expect(calls).toEqual(['root.a', 'root.b']);
    });
  });
}
