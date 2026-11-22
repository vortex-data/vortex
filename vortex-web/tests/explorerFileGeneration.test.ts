// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it } from 'bun:test';
import type { ArrayEncodingNode, LayoutTreeNode } from '../src/components/swimlane/types';
import { ExplorerSession } from '../src/explorerSession';
import { VortexWorker } from '../src/workers/VortexWorker';
import {
  createVortexWorkerHandler,
  type VortexWorkerFileHandle,
  type WorkerRequest,
  type WorkerResponse,
} from '../src/workers/workerRuntime';

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

function makeFlatTree(label: string): LayoutTreeNode {
  return {
    id: 'root',
    name: label,
    encoding: 'vortex.flat',
    dtype: 'i64',
    rowCount: 10,
    rowOffset: 0,
    segmentIds: [0],
    totalBytes: 100,
    metadataBytes: 16,
    children: [],
    childType: { kind: 'root' },
  };
}

function makeArrayTree(encoding: string): ArrayEncodingNode {
  return {
    name: 'root',
    encoding,
    dtype: 'i64',
    rowCount: 10,
    nbytes: 80,
    metadataBytes: 0,
    bufferLengths: [80],
    children: [],
  };
}

interface FakeHandle extends VortexWorkerFileHandle {
  readonly label: string;
  freeCount: number;
}

function makeFakeHandle(
  label: string,
  opts?: {
    fetchEncodingTree?: (nodeId: string) => Promise<string>;
    fetchArrayBuffer?: () => Promise<Uint8Array>;
  },
): FakeHandle {
  const handle: FakeHandle = {
    label,
    freeCount: 0,
    row_count: 10n,
    dtype: `dtype(${label})`,
    layout_tree: () => JSON.stringify(makeFlatTree(label)),
    segment_map: () => JSON.stringify([]),
    file_structure: () =>
      JSON.stringify({
        version: 1,
        eof: { version: 1, postscriptSize: 0 },
        postscript: {
          dtypeSegment: null,
          layoutSegment: 0,
          statisticsSegment: null,
          footerSegment: 0,
        },
        segments: [],
      }),
    fetch_encoding_tree:
      opts?.fetchEncodingTree ?? (async () => JSON.stringify(makeArrayTree(`enc.${label}`))),
    fetch_array_buffer: opts?.fetchArrayBuffer ?? (async () => new Uint8Array([1, 2, 3])),
    preview_array_data: async () => new Uint8Array([]),
    preview_data: async () => new Uint8Array([]),
    free() {
      handle.freeCount++;
    },
  };
  return handle;
}

function createHarness(openImpl: (file: File) => Promise<VortexWorkerFileHandle>) {
  let onmessage: ((e: MessageEvent<WorkerResponse>) => void) | null = null;
  const handler = createVortexWorkerHandler({
    init: async () => {},
    openVortexFile: openImpl,
    postMessage: (msg) => {
      onmessage?.({ data: msg } as MessageEvent<WorkerResponse>);
    },
  });

  const fakeWorker = {
    get onmessage() {
      return onmessage;
    },
    set onmessage(fn: ((e: MessageEvent<WorkerResponse>) => void) | null) {
      onmessage = fn;
    },
    postMessage(msg: WorkerRequest) {
      void handler(msg);
    },
    terminate() {},
  } as unknown as Worker;

  const client = new VortexWorker(fakeWorker);
  const session = new ExplorerSession(client);
  return { client, session };
}

describe('Explorer file generation and handle lifecycle (rank 18)', () => {
  it('keeps the newer file active and disposes the stale handle once when an older open resolves late', async () => {
    const openA = deferred<FakeHandle>();
    const openB = deferred<FakeHandle>();
    const handleA = makeFakeHandle('file-a');
    const handleB = makeFakeHandle('file-b');

    const { session } = createHarness((file) =>
      file.name === 'a.vortex' ? openA.promise : openB.promise,
    );

    const pA = session.openFile(new File(['a'], 'a.vortex'));
    // Let openA finish WASM init and enter openVortexFile before openB arrives.
    await new Promise((r) => setTimeout(r, 0));
    const pB = session.openFile(new File(['b'], 'b.vortex'));

    // Newer open completes first.
    openB.resolve(handleB);
    await pB;

    expect(session.getSnapshot().loading).toBe(false);
    expect(session.getSnapshot().error).toBeNull();
    expect(session.getSnapshot().fileState?.fileName).toBe('b.vortex');
    expect(session.getSnapshot().fileState?.layoutTree.name).toBe('file-b');
    const genB = session.getSnapshot().activeGeneration;
    expect(genB).toBeGreaterThan(0);

    // Older open resolves late: must not overwrite UI or worker handle, and must dispose handleA once.
    openA.resolve(handleA);
    await pA;

    expect(handleA.freeCount).toBe(1);
    expect(handleB.freeCount).toBe(0);
    expect(session.getSnapshot().fileState?.fileName).toBe('b.vortex');
    expect(session.getSnapshot().fileState?.layoutTree.name).toBe('file-b');
    expect(session.getSnapshot().activeGeneration).toBe(genB);
    expect(session.getSnapshot().loading).toBe(false);
    expect(session.getSnapshot().error).toBeNull();

    // Subsequent worker operation still uses handleB.
    const enc = await session.fetchEncodingTree('root');
    expect(enc.encoding).toBe('enc.file-b');
  });

  it('ignores a late error from an older open without clearing loading or overwriting a newer file', async () => {
    const openA = deferred<FakeHandle>();
    const openB = deferred<FakeHandle>();
    const handleB = makeFakeHandle('file-b');

    const { session } = createHarness((file) =>
      file.name === 'a.vortex' ? openA.promise : openB.promise,
    );

    const pA = session.openFile(new File(['a'], 'a.vortex'));
    await new Promise((r) => setTimeout(r, 0));
    const pB = session.openFile(new File(['b'], 'b.vortex'));

    // Older open fails while newer open is still in flight: busy flag must stay true, error null.
    openA.reject(new Error('corrupt file A'));
    await pA;

    expect(session.getSnapshot().loading).toBe(true);
    expect(session.getSnapshot().error).toBeNull();

    // Newer open succeeds cleanly.
    openB.resolve(handleB);
    await pB;

    expect(session.getSnapshot().loading).toBe(false);
    expect(session.getSnapshot().error).toBeNull();
    expect(session.getSnapshot().fileState?.fileName).toBe('b.vortex');
  });

  it('prevents a delayed array expansion from mutating a matching node in a replacement file and advances generation', async () => {
    const expandA = deferred<string>();
    const handleA = makeFakeHandle('file-a', {
      fetchEncodingTree: () => expandA.promise,
    });
    const handleB = makeFakeHandle('file-b');

    const { session } = createHarness(async (file) =>
      file.name === 'a.vortex' ? handleA : handleB,
    );

    await session.openFile(new File(['a'], 'a.vortex'));
    const genA = session.getSnapshot().activeGeneration;

    // Start expanding node 'root' on file A.
    const expandPromise = session.expandArrayTree('root', genA);

    // Replace with file B (which has the same 'root' node ID) before expansion finishes.
    await session.openFile(new File(['b'], 'b.vortex'));
    const genB = session.getSnapshot().activeGeneration;

    // Generation incremented so SelectionProvider keyed on activeGeneration resets selection.
    expect(genB).toBeGreaterThan(genA);
    // Handle A is still in flight for expandA, so it must NOT be freed yet.
    expect(handleA.freeCount).toBe(0);

    // Complete delayed expansion from file A.
    expandA.resolve(JSON.stringify(makeArrayTree('enc.stale-a')));
    await expandPromise;

    // Handle A is disposed exactly once after its outstanding use finishes.
    expect(handleA.freeCount).toBe(1);
    expect(handleB.freeCount).toBe(0);

    // File B's 'root' node must not have received file A's array children.
    expect(session.getSnapshot().fileState?.fileName).toBe('b.vortex');
    expect(session.getSnapshot().fileState?.layoutTree.children).toEqual([]);
    expect(session.getSnapshot().fileState?.layoutTree.arrayEncodingTree).toBeUndefined();
  });

  it('invalidates a pending open on close and disposes the handle once when it resolves', async () => {
    const openA = deferred<FakeHandle>();
    const handleA = makeFakeHandle('file-a');

    const { session } = createHarness(() => openA.promise);

    const pA = session.openFile(new File(['a'], 'a.vortex'));
    await new Promise((r) => setTimeout(r, 0));
    expect(session.getSnapshot().loading).toBe(true);

    session.closeFile();
    expect(session.getSnapshot().loading).toBe(false);
    expect(session.getSnapshot().fileState).toBeNull();
    expect(session.getSnapshot().error).toBeNull();

    openA.resolve(handleA);
    await pA;

    expect(handleA.freeCount).toBe(1);
    expect(session.getSnapshot().fileState).toBeNull();
    expect(session.getSnapshot().loading).toBe(false);
  });

  it('disposes the active handle once on close after an in-flight read completes', async () => {
    const readBuf = deferred<Uint8Array>();
    const handleA = makeFakeHandle('file-a', {
      fetchArrayBuffer: () => readBuf.promise,
    });

    const { session } = createHarness(async () => handleA);

    await session.openFile(new File(['a'], 'a.vortex'));
    const genA = session.getSnapshot().activeGeneration;

    const bufPromise = session.fetchArrayBuffer('root', [], 0, genA);
    session.closeFile();

    // Still in flight, so not freed yet.
    expect(handleA.freeCount).toBe(0);

    readBuf.resolve(new Uint8Array([42]));
    await expect(bufPromise).rejects.toThrow('Stale file generation');

    // Freed exactly once after the in-flight operation finishes.
    expect(handleA.freeCount).toBe(1);

    // Repeated close does not double-free.
    session.closeFile();
    expect(handleA.freeCount).toBe(1);
  });
});
