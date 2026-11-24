// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterEach, describe, expect, mock, test } from 'bun:test';

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

interface FakeHandle {
  row_count: bigint;
  dtype: string;
  freeCount: number;
  layout_tree: () => string;
  segment_map: () => string;
  file_structure: () => string;
  fetch_encoding_tree: (nodeId: string) => Promise<string>;
  fetch_array_buffer: () => Promise<Uint8Array>;
  preview_array_data: () => Promise<Uint8Array>;
  preview_data: () => Promise<Uint8Array>;
  free: () => void;
}

function makeHandle(
  name: string,
  fetchEncodingTree: (nodeId: string) => Promise<string> = async () => JSON.stringify({ name }),
): FakeHandle {
  const handle: FakeHandle = {
    row_count: 1n,
    dtype: 'i32',
    freeCount: 0,
    layout_tree: () => JSON.stringify({ id: 'layout-0', name }),
    segment_map: () => '[]',
    file_structure: () => JSON.stringify({ version: 1 }),
    fetch_encoding_tree: fetchEncodingTree,
    fetch_array_buffer: async () => new Uint8Array([1]),
    preview_array_data: async () => new Uint8Array([1]),
    preview_data: async () => new Uint8Array([1]),
    free() {
      handle.freeCount++;
    },
  };
  return handle;
}

type WorkerReply =
  | { type: 'result'; id: number; data: unknown }
  | { type: 'error'; id: number; error: string };

const replies: WorkerReply[] = [];
let openHandler: (file: File) => Promise<FakeHandle> = async () => makeHandle('default');

const fakeSelf = {
  onmessage: null as ((e: MessageEvent) => Promise<void>) | null,
  postMessage(msg: WorkerReply) {
    replies.push(msg);
  },
};

Object.assign(globalThis, { self: fakeSelf });

mock.module('../src/wasm/pkg/vortex_web_wasm.js', () => ({
  default: async () => {},
  open_vortex_file: (file: File) => openHandler(file),
}));

await import('../src/workers/vortex.worker.ts');

async function send(data: Record<string, unknown>): Promise<void> {
  await fakeSelf.onmessage?.({ data } as MessageEvent);
}

afterEach(async () => {
  await send({ type: 'close' });
  replies.length = 0;
});

describe('vortex.worker handle lifecycle', () => {
  test('stale open frees its own handle and does not replace a newer open', async () => {
    const openA = deferred<FakeHandle>();
    const openB = deferred<FakeHandle>();
    openHandler = (file) => (file.name === 'a.vortex' ? openA.promise : openB.promise);

    const pA = send({ type: 'open', id: 1, file: new File(['a'], 'a.vortex') });
    const pB = send({ type: 'open', id: 2, file: new File(['b'], 'b.vortex') });

    const handleB = makeHandle('B');
    openB.resolve(handleB);
    await pB;

    const handleA = makeHandle('A');
    openA.resolve(handleA);
    await pA;

    expect(handleA.freeCount).toBe(1);
    expect(handleB.freeCount).toBe(0);
    expect(replies.find((r) => r.id === 2)).toMatchObject({ type: 'result' });
    expect(replies.find((r) => r.id === 1)).toEqual({
      type: 'error',
      id: 1,
      error: 'Stale file open',
    });

    await send({ type: 'fetchEncodingTree', id: 3, nodeId: 'layout-0' });
    expect(replies.find((r) => r.id === 3)).toEqual({
      type: 'result',
      id: 3,
      data: { name: 'B' },
    });
  });

  test('keeps active handle readable during replacement open and defers free until reads finish', async () => {
    const readA = deferred<string>();
    const handleA = makeHandle('A', () => readA.promise);
    const openB = deferred<FakeHandle>();
    openHandler = (file) => (file.name === 'a.vortex' ? Promise.resolve(handleA) : openB.promise);

    await send({ type: 'open', id: 1, file: new File(['a'], 'a.vortex') });

    const pOpenB = send({ type: 'open', id: 2, file: new File(['b'], 'b.vortex') });
    const pReadA = send({ type: 'fetchEncodingTree', id: 3, nodeId: 'layout-0' });
    await Promise.resolve();

    const handleB = makeHandle('B');
    openB.resolve(handleB);
    await pOpenB;

    expect(handleA.freeCount).toBe(0);
    expect(handleB.freeCount).toBe(0);

    readA.resolve(JSON.stringify({ name: 'A-tree' }));
    await pReadA;

    expect(handleA.freeCount).toBe(1);
    expect(replies.find((r) => r.id === 3)).toEqual({
      type: 'result',
      id: 3,
      data: { name: 'A-tree' },
    });
  });

  test('close frees active handle after in-flight reads and discards in-flight opens', async () => {
    const readA = deferred<string>();
    const handleA = makeHandle('A', () => readA.promise);
    const openB = deferred<FakeHandle>();
    openHandler = (file) => (file.name === 'a.vortex' ? Promise.resolve(handleA) : openB.promise);

    await send({ type: 'open', id: 1, file: new File(['a'], 'a.vortex') });
    const pReadA = send({ type: 'fetchEncodingTree', id: 2, nodeId: 'layout-0' });
    const pOpenB = send({ type: 'open', id: 3, file: new File(['b'], 'b.vortex') });
    await Promise.resolve();

    await send({ type: 'close' });
    expect(handleA.freeCount).toBe(0);

    readA.resolve(JSON.stringify({ name: 'A-tree' }));
    await pReadA;
    expect(handleA.freeCount).toBe(1);

    const handleB = makeHandle('B');
    openB.resolve(handleB);
    await pOpenB;
    expect(handleB.freeCount).toBe(1);
    expect(replies.find((r) => r.id === 3)).toEqual({
      type: 'error',
      id: 3,
      error: 'Stale file open',
    });

    await send({ type: 'fetchEncodingTree', id: 4, nodeId: 'layout-0' });
    expect(replies.find((r) => r.id === 4)).toMatchObject({ type: 'error', id: 4 });
  });
});
