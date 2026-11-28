// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, test } from 'bun:test';
import { VortexWorker, type WorkerLike } from '../src/workers/VortexWorker';

class FakeWorker implements WorkerLike {
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  posted: Array<Record<string, unknown>> = [];
  terminated = false;

  postMessage(message: unknown): void {
    this.posted.push(message as Record<string, unknown>);
  }

  terminate(): void {
    this.terminated = true;
  }

  emitMessage(data: unknown): void {
    this.onmessage?.({ data } as MessageEvent);
  }

  emitError(message: string): void {
    this.onerror?.({ message } as ErrorEvent);
  }
}

describe('VortexWorker request settlement', () => {
  test('rejects all pending requests and future requests on worker error and terminates the worker', async () => {
    const fake = new FakeWorker();
    const client = new VortexWorker(fake);
    expect(client.isClosed).toBe(false);

    const pA = client.fetchEncodingTree('node-a');
    const pB = client.previewData('node-b', 10);

    const idA = fake.posted[0].id as number;
    const idB = fake.posted[1].id as number;

    fake.emitError('WASM unreachable');
    expect(fake.terminated).toBe(true);
    expect(client.isClosed).toBe(true);
    // Late replies and duplicate errors after terminal failure must be ignored.
    fake.emitMessage({ type: 'result', id: idA, data: {} });
    fake.emitMessage({ type: 'error', id: idB, error: 'late' });
    fake.emitError('duplicate');

    await expect(pA).rejects.toThrow('WASM unreachable');
    await expect(pB).rejects.toThrow('WASM unreachable');
    await expect(client.previewData('node-c', 5)).rejects.toThrow('WASM unreachable');
  });

  test('rejects all pending requests and terminates worker on fatal message', async () => {
    const fake = new FakeWorker();
    const client = new VortexWorker(fake);

    const pA = client.fetchEncodingTree('node-a');
    fake.emitMessage({ type: 'fatal', error: 'RuntimeError: unreachable' });

    expect(fake.terminated).toBe(true);
    expect(client.isClosed).toBe(true);
    await expect(pA).rejects.toThrow('Worker error: RuntimeError: unreachable');
    await expect(client.previewData('node-b', 5)).rejects.toThrow(
      'Worker error: RuntimeError: unreachable',
    );
  });

  test('closeFile posts only while open and is a no-op once closed or terminated', () => {
    const fake1 = new FakeWorker();
    const client1 = new VortexWorker(fake1);
    client1.closeFile();
    expect(fake1.posted).toEqual([{ type: 'close' }]);

    fake1.emitError('fatal crash');
    client1.closeFile();
    expect(fake1.posted).toEqual([{ type: 'close' }]);

    const fake2 = new FakeWorker();
    const client2 = new VortexWorker(fake2);
    client2.terminate();
    expect(client2.isClosed).toBe(true);
    client2.closeFile();
    expect(fake2.posted).toEqual([]);
  });

  test('rejects cleanly if postMessage throws synchronously', async () => {
    const fake = new FakeWorker();
    fake.postMessage = () => {
      throw new Error('Clone failed');
    };
    const client = new VortexWorker(fake);

    await expect(client.fetchEncodingTree('node-a')).rejects.toThrow('Clone failed');
    expect(client.isClosed).toBe(false);
  });

  test('rejects all pending requests on terminate and ignores late replies', async () => {
    const fake = new FakeWorker();
    const client = new VortexWorker(fake);

    const pA = client.fetchEncodingTree('node-a');
    const pB = client.previewArrayData('node-b', ['values'], 10);
    const idA = fake.posted[0].id as number;
    const idB = fake.posted[1].id as number;

    client.terminate();
    expect(fake.terminated).toBe(true);

    // Late replies after termination must not throw.
    fake.emitMessage({ type: 'result', id: idA, data: {} });
    fake.emitMessage({ type: 'result', id: idB, data: new Uint8Array([1]) });

    await expect(pA).rejects.toThrow('Worker terminated');
    await expect(pB).rejects.toThrow('Worker terminated');
    await expect(client.previewData('node-c', 1)).rejects.toThrow('Worker terminated');
  });

  test('keeps other pending and subsequent requests usable after an ordinary application error', async () => {
    const fake = new FakeWorker();
    const client = new VortexWorker(fake);

    const pErr = client.fetchEncodingTree('bad-node');
    const pOk = client.previewData('good-node', 5);
    const errId = fake.posted[0].id as number;
    const okId = fake.posted[1].id as number;

    fake.emitMessage({ type: 'error', id: errId, error: 'Node not found' });
    await expect(pErr).rejects.toThrow('Node not found');

    const bytes = new Uint8Array([42]);
    fake.emitMessage({ type: 'result', id: okId, data: bytes });
    expect(await pOk).toEqual(bytes);
  });
});
