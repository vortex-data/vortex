// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterEach, describe, expect, test } from 'bun:test';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { DetailPanel } from '../src/components/detail/DetailPanel';
import { TreePanel } from '../src/components/explorer/TreePanel';
import { SwimlaneOverview } from '../src/components/swimlane/SwimlaneOverview';
import type { LayoutTreeNode } from '../src/components/swimlane/types';
import { SelectionContext, type SelectionContextValue } from '../src/contexts/SelectionContextCore';
import type { VortexFileContextValue } from '../src/contexts/VortexFileContext';
import { VortexFileContext } from '../src/contexts/VortexFileContextCore';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

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
  selection: SelectionContextValue = emptySelection,
): Promise<HTMLDivElement> {
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  mountedRoot = root;
  mountedContainer = container;
  await act(async () => {
    root.render(
      <VortexFileContext.Provider value={file}>
        <SelectionContext.Provider value={selection}>{element}</SelectionContext.Provider>
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

    test('retries on the next expansion when collapsed before a pending read rejects', async () => {
      let calls = 0;
      let rejectFirst!: (err: Error) => void;
      let unhandled = 0;
      const onUnhandled = () => {
        unhandled++;
      };
      process.on('unhandledRejection', onUnhandled);
      try {
        const file = makeFlatFile(() => {
          calls++;
          if (calls === 1) {
            return new Promise<void>((_, reject) => {
              rejectFirst = reject;
            });
          }
          return Promise.resolve();
        });

        const container = await mountView(renderView(), file);
        expect(calls).toBe(1);

        const toggle = () =>
          act(async () => {
            const btn = container.querySelector(toggleSelector);
            expect(btn).not.toBeNull();
            btn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
          });

        // Collapse while pending, then re-expand while still pending: must stay at one request.
        await toggle();
        await toggle();
        expect(calls).toBe(1);

        // Collapse while still pending, let the request reject, then expand once: must start a second read.
        await toggle();
        rejectFirst(new Error('temporary'));
        await flushMicrotasks();
        expect(calls).toBe(1);
        expect(unhandled).toBe(0);

        await toggle();
        await flushMicrotasks();
        expect(calls).toBe(2);
        expect(unhandled).toBe(0);
      } finally {
        process.off('unhandledRejection', onUnhandled);
      }
    });
  });
}

describe('DetailPanel array tree expansion', () => {
  test('catches rejected expandArrayTree calls from BlockTreemap', async () => {
    let calls = 0;
    let unhandled = 0;
    const onUnhandled = () => {
      unhandled++;
    };
    process.on('unhandledRejection', onUnhandled);
    try {
      const file = makeFlatFile(() => {
        calls++;
        return Promise.reject(new Error('temporary'));
      });
      const selection: SelectionContextValue = {
        state: {
          selectedNode: file.layoutTree,
          selectedNodeId: file.layoutTree.id,
          hoveredNodeId: null,
        },
        selectNode() {},
        hoverNode() {},
      };

      const container = await mountView(<DetailPanel />, file, selection);
      const treemapTab = Array.from(container.querySelectorAll('button[role="tab"]')).find(
        (btn) => btn.textContent === 'Treemap',
      );
      expect(treemapTab).not.toBeUndefined();

      await act(async () => {
        treemapTab!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      });
      await flushMicrotasks();

      expect(calls).toBe(1);
      expect(unhandled).toBe(0);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });
});
