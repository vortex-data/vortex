// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, test } from 'bun:test';
import { VortexWorker, type WorkerLike } from '../src/workers/VortexWorker';

class FakeWorker implements WorkerLike {
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  onmessageerror: ((e: MessageEvent) => void) | null = null;
  posted: Array<Record<string, unknown>> = [];
  terminated = false;
  postMessageImpl?: (message: unknown) => void;

  postMessage(message: unknown): void {
    if (this.postMessageImpl) {
      this.postMessageImpl(message);
    }
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

  emitMessageError(): void {
    this.onmessageerror?.({} as MessageEvent);
  }
}

describe('VortexWorker request settlement', () => {
  test('rejects all pending requests and future requests on worker error, settling each once', async () => {
    const fake = new FakeWorker();
    const client = new VortexWorker(fake);

    let settleA = 0;
    let settleB = 0;
    const pA = client.fetchEncodingTree('node-a').then(
      () => {
        settleA++;
        return 'resolved';
      },
      (err: Error) => {
        settleA++;
        return err.message;
      },
    );
    const pB = client.previewData('node-b', 10).then(
      () => {
        settleB++;
        return 'resolved';
      },
      (err: Error) => {
        settleB++;
        return err.message;
      },
    );

    const idA = fake.posted[0].id as number;
    const idB = fake.posted[1].id as number;

    fake.emitError('WASM unreachable');
    // Late replies and duplicate errors after terminal failure must be ignored.
    fake.emitMessage({ type: 'result', id: idA, data: {} });
    fake.emitMessage({ type: 'error', id: idB, error: 'late' });
    fake.emitError('duplicate');

    expect(await pA).toContain('WASM unreachable');
    expect(await pB).toContain('WASM unreachable');
    expect(settleA).toBe(1);
    expect(settleB).toBe(1);

    await expect(client.previewData('node-c', 5)).rejects.toThrow('WASM unreachable');
  });

  test('rejects all pending requests and future requests on worker messageerror', async () => {
    const fake = new FakeWorker();
    const client = new VortexWorker(fake);

    const pA = client.fetchEncodingTree('node-a');
    const pB = client.fetchArrayBuffer('node-b', [], 0);

    fake.emitMessageError();

    await expect(pA).rejects.toThrow('Worker message deserialization failed');
    await expect(pB).rejects.toThrow('Worker message deserialization failed');
    await expect(client.fetchEncodingTree('node-c')).rejects.toThrow(
      'Worker message deserialization failed',
    );
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

    // Late replies after termination must not throw or double-settle.
    fake.emitMessage({ type: 'result', id: idA, data: {} });
    fake.emitMessage({ type: 'result', id: idB, data: new Uint8Array([1]) });

    await expect(pA).rejects.toThrow('Worker terminated');
    await expect(pB).rejects.toThrow('Worker terminated');
    await expect(client.previewData('node-c', 1)).rejects.toThrow('Worker terminated');
  });

  test('rejects and cleans up a request when synchronous postMessage throws without poisoning other requests', async () => {
    const fake = new FakeWorker();
    const client = new VortexWorker(fake);

    const pOk = client.previewData('node-ok', 5);
    const okId = fake.posted[0].id as number;

    fake.postMessageImpl = () => {
      throw new Error('DataCloneError');
    };
    const pFail = client.previewData('node-bad', 5);
    fake.postMessageImpl = undefined;

    await expect(pFail).rejects.toThrow('DataCloneError');

    const payload = new Uint8Array([1, 2, 3]);
    fake.emitMessage({ type: 'result', id: okId, data: payload });
    expect(await pOk).toEqual(payload);

    // Subsequent requests still work.
    const pAfter = client.previewData('node-after', 5);
    const afterId = fake.posted[1].id as number;
    fake.emitMessage({ type: 'result', id: afterId, data: new Uint8Array([9]) });
    expect(await pAfter).toEqual(new Uint8Array([9]));
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
