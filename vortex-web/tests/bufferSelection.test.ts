// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, test } from 'bun:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { BuffersPane } from '../src/components/detail/BuffersPane';
import { BufferSelectionController } from '../src/components/detail/bufferSelection';
import type { LayoutTreeNode } from '../src/components/swimlane/types';
import type { VortexFileContextValue } from '../src/contexts/VortexFileContext';
import { VortexFileContext } from '../src/contexts/VortexFileContextCore';

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

describe('BufferSelectionController', () => {
  test('keeps bytes from the second buffer when the first read resolves late', async () => {
    const read0 = deferred<Uint8Array>();
    const read1 = deferred<Uint8Array>();

    const controller = new BufferSelectionController((_layoutNodeId, _arrayPath, bufferIndex) =>
      bufferIndex === 0 ? read0.promise : read1.promise,
    );

    const p0 = controller.selectNode('root.0::arr.0', 2);
    const p1 = controller.selectBuffer(1);

    const bytes1 = new Uint8Array([0xbe, 0xef]);
    read1.resolve(bytes1);
    await p1;

    expect(controller.getSnapshot().selectedBuffer).toBe(1);
    expect(controller.getSnapshot().loading).toBe(false);
    expect(controller.getSnapshot().error).toBeNull();
    expect(controller.getSnapshot().bufferData).toEqual(bytes1);

    // Late resolution for buffer 0 must not overwrite buffer 1 bytes.
    read0.resolve(new Uint8Array([0xde, 0xad]));
    await p0;

    expect(controller.getSnapshot().selectedBuffer).toBe(1);
    expect(controller.getSnapshot().loading).toBe(false);
    expect(controller.getSnapshot().error).toBeNull();
    expect(controller.getSnapshot().bufferData).toEqual(bytes1);
  });

  test('ignores a late failure from a superseded read without clearing loading or newer bytes', async () => {
    const read0 = deferred<Uint8Array>();
    const read1 = deferred<Uint8Array>();

    const controller = new BufferSelectionController((_layoutNodeId, _arrayPath, bufferIndex) =>
      bufferIndex === 0 ? read0.promise : read1.promise,
    );

    const p0 = controller.selectNode('root.0::arr.0', 2);
    const p1 = controller.selectBuffer(1);

    // Older read fails while newer read is still in flight.
    read0.reject(new Error('stale buffer read failed'));
    await p0;

    expect(controller.getSnapshot().selectedBuffer).toBe(1);
    expect(controller.getSnapshot().loading).toBe(true);
    expect(controller.getSnapshot().error).toBeNull();

    const bytes1 = new Uint8Array([0x01, 0x02, 0x03]);
    read1.resolve(bytes1);
    await p1;

    expect(controller.getSnapshot().loading).toBe(false);
    expect(controller.getSnapshot().error).toBeNull();
    expect(controller.getSnapshot().bufferData).toEqual(bytes1);
  });

  test('ignores a stale read when switching to a new node reusing buffer index 0', async () => {
    const readNodeA = deferred<Uint8Array>();
    const readNodeB = deferred<Uint8Array>();

    const controller = new BufferSelectionController((layoutNodeId, arrayPath, bufferIndex) => {
      expect(bufferIndex).toBe(0);
      expect(arrayPath).toEqual(['values']);
      return layoutNodeId === 'root.a' ? readNodeA.promise : readNodeB.promise;
    });

    const pA = controller.selectNode('root.a.$array.$values', 1);
    const pB = controller.selectNode('root.b.$array.$values', 1);

    const bytesB = new Uint8Array([0x22, 0x33]);
    readNodeB.resolve(bytesB);
    await pB;

    readNodeA.resolve(new Uint8Array([0x11, 0x11]));
    await pA;

    expect(controller.getSnapshot().selectedBuffer).toBe(0);
    expect(controller.getSnapshot().loading).toBe(false);
    expect(controller.getSnapshot().bufferData).toEqual(bytesB);
  });

  test('clears selection and ignores in-flight reads when switching to a node with zero buffers', async () => {
    const readNodeA = deferred<Uint8Array>();

    const controller = new BufferSelectionController(() => readNodeA.promise);

    const pA = controller.selectNode('root.a::arr.0', 1);
    expect(controller.getSnapshot().loading).toBe(true);

    await controller.selectNode('root.empty::arr.0', 0);
    expect(controller.getSnapshot()).toEqual({
      nodeId: 'root.empty::arr.0',
      selectedBuffer: null,
      bufferData: null,
      loading: false,
      error: null,
    });

    readNodeA.resolve(new Uint8Array([0xaa]));
    await pA;

    expect(controller.getSnapshot()).toEqual({
      nodeId: 'root.empty::arr.0',
      selectedBuffer: null,
      bufferData: null,
      loading: false,
      error: null,
    });
  });

  test('ignores late completions after unmount invalidation', async () => {
    const read0 = deferred<Uint8Array>();
    let notifications = 0;

    const controller = new BufferSelectionController(() => read0.promise);
    controller.subscribe(() => {
      notifications += 1;
    });

    const p0 = controller.selectNode('root.a::arr.0', 1);
    const notificationsBeforeUnmount = notifications;

    controller.invalidate();
    read0.resolve(new Uint8Array([0xff]));
    await p0;

    expect(notifications).toBe(notificationsBeforeUnmount);
    expect(controller.getSnapshot().bufferData).toBeNull();
  });

  test('suppresses previous node bytes on the first committed render after node changes', async () => {
    const fetchArrayBuffer = async (layoutNodeId: string) =>
      layoutNodeId === 'root.a' ? new Uint8Array([0x11]) : new Uint8Array([0x22]);

    const controller = new BufferSelectionController(fetchArrayBuffer);
    const fileContext = { fetchArrayBuffer } as unknown as VortexFileContextValue;

    const makeNode = (id: string, bufferName: string): LayoutTreeNode => ({
      id,
      name: id,
      encoding: 'vortex.primitive',
      dtype: 'u8',
      rowCount: 1,
      rowOffset: 0,
      segmentIds: [],
      metadataBytes: 0,
      totalBytes: 1,
      bufferLengths: [1],
      bufferNames: [bufferName],
      children: [],
    });

    const nodeA = makeNode('root.a.$array.$values', 'buf-a');
    const nodeB = makeNode('root.b.$array.$values', 'buf-b');

    // Load node A so the controller snapshot holds A's byte 0x11.
    await controller.selectNode(nodeA.id, 1);
    const htmlA = renderToStaticMarkup(
      createElement(
        VortexFileContext.Provider,
        { value: fileContext },
        createElement(BuffersPane, { node: nodeA, controller }),
      ),
    );
    expect(htmlA).toContain('buf-a');
    expect(htmlA).toContain('>11<');

    // Switch BuffersPane to node B before useEffect runs controller.selectNode(nodeB.id, 1).
    // The first committed render must show B's label without A's stale byte 11.
    const htmlBFirstRender = renderToStaticMarkup(
      createElement(
        VortexFileContext.Provider,
        { value: fileContext },
        createElement(BuffersPane, { node: nodeB, controller }),
      ),
    );
    expect(htmlBFirstRender).toContain('buf-b');
    expect(htmlBFirstRender).not.toContain('>11<');

    // After the effect loads node B, B's own byte 0x22 is shown.
    await controller.selectNode(nodeB.id, 1);
    const htmlBLoaded = renderToStaticMarkup(
      createElement(
        VortexFileContext.Provider,
        { value: fileContext },
        createElement(BuffersPane, { node: nodeB, controller }),
      ),
    );
    expect(htmlBLoaded).toContain('buf-b');
    expect(htmlBLoaded).toContain('>22<');
  });
});
