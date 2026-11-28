// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterEach, describe, expect, test } from 'bun:test';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import App from '../src/App';
import { ThemeContext, type ThemeContextValue } from '../src/contexts/ThemeContextCore';
import type { OpenFileResult, WorkerLike } from '../src/workers/VortexWorker';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const themeValue: ThemeContextValue = {
  choice: 'system',
  resolved: 'light',
  setChoice() {},
};

function makeOpenResult(rowCount = 42): OpenFileResult {
  return {
    rowCount,
    dtype: 'i64',
    layoutTree: {
      id: 'root',
      encoding: 'vortex.struct',
      dtype: '{a=i64}',
      rowCount,
      rowOffset: 0,
      metadataBytes: 64,
      segmentIds: [],
      childType: { kind: 'root' },
      children: [],
    },
    segments: [
      {
        index: 0,
        byteOffset: 0,
        byteLength: 256,
        alignment: 64,
        column: null,
        layoutPath: 'root',
      },
    ],
    fileStructure: {
      fileSize: 512,
      version: 1,
      postscriptSize: 64,
      totalDataBytes: 256,
      totalMetadataBytes: 128,
    },
  };
}

interface PostedMessage {
  type: string;
  id?: number;
  file?: File;
  [key: string]: unknown;
}

class StubWorker implements WorkerLike {
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  posted: PostedMessage[] = [];
  postedAfterTerminate: PostedMessage[] = [];
  terminated = false;

  private readonly onPost: (worker: StubWorker, msg: PostedMessage) => void;

  constructor(onPost: (worker: StubWorker, msg: PostedMessage) => void) {
    this.onPost = onPost;
  }

  postMessage(message: unknown): void {
    const msg = message as PostedMessage;
    if (this.terminated) {
      this.postedAfterTerminate.push(msg);
      return;
    }
    this.posted.push(msg);
    this.onPost(this, msg);
  }

  terminate(): void {
    this.terminated = true;
  }

  emitResult(id: number, data: unknown): void {
    this.onmessage?.({ data: { type: 'result', id, data } } as MessageEvent);
  }

  emitRequestError(id: number, error: string): void {
    this.onmessage?.({ data: { type: 'error', id, error } } as MessageEvent);
  }

  emitFatalError(message: string): void {
    this.onerror?.({ message } as ErrorEvent);
  }
}

const originalWorker = globalThis.Worker;
let mountedRoot: Root | null = null;
let mountedContainer: HTMLDivElement | null = null;

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
  globalThis.Worker = originalWorker;
});

async function mountApp(): Promise<HTMLDivElement> {
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  mountedRoot = root;
  mountedContainer = container;
  await act(async () => {
    root.render(
      <ThemeContext.Provider value={themeValue}>
        <App />
      </ThemeContext.Provider>,
    );
  });
  return container;
}

async function chooseFile(container: HTMLElement, name: string): Promise<void> {
  const input = container.querySelector<HTMLInputElement>('input[type="file"]');
  expect(input).not.toBeNull();
  Object.defineProperty(input!, 'files', {
    configurable: true,
    value: [new File(['sample'], name)],
  });
  await act(async () => {
    input!.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

async function dropFileOnApp(container: HTMLElement, name: string): Promise<void> {
  const app = container.querySelector('.x-app');
  expect(app).not.toBeNull();
  const dropEvent = new Event('drop', { bubbles: true, cancelable: true });
  Object.defineProperty(dropEvent, 'dataTransfer', {
    value: { files: [new File(['sample'], name)] },
  });
  await act(async () => {
    app!.dispatchEvent(dropEvent);
  });
}

describe('App worker recovery', () => {
  test('recreates worker after fatal crash and keeps live worker reusable after ordinary errors', async () => {
    const workers: StubWorker[] = [];
    const openedFiles: string[] = [];

    const FakeWorkerCtor = class {
      constructor() {
        const idx = workers.length;
        const stub = new StubWorker((w, msg) => {
          if (msg.type !== 'open' || !msg.file || msg.id === undefined) return;
          openedFiles.push(msg.file.name);
          if (idx === 0) {
            queueMicrotask(() => w.emitFatalError('WASM unreachable'));
          } else if (msg.file.name === 'bad.vortex') {
            queueMicrotask(() => w.emitRequestError(msg.id!, 'Invalid Vortex magic bytes'));
          } else {
            queueMicrotask(() => w.emitResult(msg.id!, makeOpenResult(100)));
          }
        });
        workers.push(stub);
        // biome-ignore lint/correctness/noConstructorReturn: stub constructor delegates to StubWorker instance.
        return stub;
      }
    };
    globalThis.Worker = FakeWorkerCtor as unknown as typeof Worker;

    const container = await mountApp();
    expect(workers).toHaveLength(1);

    // 1. Fatal worker crash during openFile.
    await chooseFile(container, 'crash.vortex');
    expect(openedFiles).toEqual(['crash.vortex']);
    expect(workers[0]!.terminated).toBe(true);
    expect(workers[0]!.postedAfterTerminate).toEqual([]);
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      'Worker error: WASM unreachable',
    );

    // 2. Next selection creates a fresh worker; ordinary error keeps it alive.
    await chooseFile(container, 'bad.vortex');
    expect(workers).toHaveLength(2);
    expect(openedFiles).toEqual(['crash.vortex', 'bad.vortex']);
    expect(workers[1]!.terminated).toBe(false);
    expect(workers[1]!.posted.some((m) => m.type === 'close')).toBe(true);
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      'Invalid Vortex magic bytes',
    );

    // 3. Subsequent selection reuses the live second worker and opens cleanly.
    await chooseFile(container, 'good.vortex');
    expect(workers).toHaveLength(2);
    expect(openedFiles).toEqual(['crash.vortex', 'bad.vortex', 'good.vortex']);
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(container.textContent).toContain('good.vortex');
  });

  test('recovers cleanly when worker crashes while a file is open before close or drop', async () => {
    const workers: StubWorker[] = [];

    const FakeWorkerCtor = class {
      constructor() {
        const stub = new StubWorker((w, msg) => {
          if (msg.type === 'open' && msg.id !== undefined) {
            queueMicrotask(() => w.emitResult(msg.id!, makeOpenResult(10)));
          }
        });
        workers.push(stub);
        // biome-ignore lint/correctness/noConstructorReturn: stub constructor delegates to StubWorker instance.
        return stub;
      }
    };
    globalThis.Worker = FakeWorkerCtor as unknown as typeof Worker;

    const container = await mountApp();
    await chooseFile(container, 'open-a.vortex');
    expect(workers).toHaveLength(1);
    expect(container.textContent).toContain('open-a.vortex');

    // Crash the active worker after open succeeded.
    await act(async () => {
      workers[0]!.emitFatalError('out of memory');
    });
    expect(workers[0]!.terminated).toBe(true);

    // Dropping a new file replaces the dead worker without posting to it.
    await dropFileOnApp(container, 'open-b.vortex');
    expect(workers).toHaveLength(2);
    expect(workers[0]!.postedAfterTerminate).toEqual([]);
    expect(container.textContent).toContain('open-b.vortex');

    // Crash the second worker, click Close, then choose a third file.
    await act(async () => {
      workers[1]!.emitFatalError('trapped');
    });
    const closeButton = Array.from(container.querySelectorAll('button')).find((b) =>
      b.textContent?.includes('Close'),
    );
    expect(closeButton).toBeDefined();
    await act(async () => {
      closeButton!.click();
    });
    expect(workers[1]!.postedAfterTerminate).toEqual([]);

    await chooseFile(container, 'open-c.vortex');
    expect(workers).toHaveLength(3);
    expect(container.textContent).toContain('open-c.vortex');
  });

  test('preserves open-file epoch guard when a stale open fails after a newer open on a replacement worker', async () => {
    const workers: StubWorker[] = [];
    const pendingOpens: Array<{ worker: StubWorker; id: number; name: string }> = [];

    const FakeWorkerCtor = class {
      constructor() {
        const stub = new StubWorker((w, msg) => {
          if (msg.type === 'open' && msg.file && msg.id !== undefined) {
            pendingOpens.push({ worker: w, id: msg.id, name: msg.file.name });
          }
        });
        workers.push(stub);
        // biome-ignore lint/correctness/noConstructorReturn: stub constructor delegates to StubWorker instance.
        return stub;
      }
    };
    globalThis.Worker = FakeWorkerCtor as unknown as typeof Worker;

    const container = await mountApp();

    // Start opening file A on worker 0; do not settle it yet.
    await chooseFile(container, 'stale.vortex');
    expect(pendingOpens).toHaveLength(1);

    // Crash worker 0 and immediately start opening file B before flushing microtasks.
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!;
    Object.defineProperty(input, 'files', {
      configurable: true,
      value: [new File(['sample'], 'fresh.vortex')],
    });
    await act(async () => {
      workers[0]!.emitFatalError('stale worker died');
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });

    expect(workers).toHaveLength(2);
    expect(pendingOpens).toHaveLength(2);
    expect(pendingOpens[1]!.name).toBe('fresh.vortex');

    // Settle file B on worker 1; stale failure from worker 0 must not clobber it.
    await act(async () => {
      pendingOpens[1]!.worker.emitResult(pendingOpens[1]!.id, makeOpenResult(77));
    });

    expect(workers[1]!.terminated).toBe(false);
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(container.textContent).toContain('fresh.vortex');
  });
});
