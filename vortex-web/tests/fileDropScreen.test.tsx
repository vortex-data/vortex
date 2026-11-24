// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterEach, describe, expect, test } from 'bun:test';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { FileDropScreen } from '../src/components/explorer/FileDropScreen';
import { ThemeContext, type ThemeContextValue } from '../src/contexts/ThemeContextCore';

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

const themeValue: ThemeContextValue = {
  choice: 'system',
  resolved: 'light',
  setChoice() {},
};

const originalFetch = globalThis.fetch;
let mountedRoot: Root | null = null;
let mountedContainer: HTMLDivElement | null = null;

afterEach(async () => {
  globalThis.fetch = originalFetch;
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

async function submitUrl(container: HTMLElement, url: string) {
  const input = container.querySelector<HTMLInputElement>('input[type="url"]');
  const form = container.querySelector<HTMLFormElement>('form');
  expect(input).not.toBeNull();
  expect(form).not.toBeNull();

  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
  await act(async () => {
    setter?.call(input, url);
    input!.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await act(async () => {
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  });
}

function DropScreenHarness({ onSelect }: { onSelect: (name: string) => void }) {
  const [fileName, setFileName] = useState('');
  return (
    <ThemeContext.Provider value={themeValue}>
      {fileName ? (
        <div data-current>{fileName}</div>
      ) : (
        <FileDropScreen
          error={null}
          loading={false}
          onFileLoaded={(file) => {
            onSelect(file.name);
            setFileName(file.name);
          }}
        />
      )}
    </ThemeContext.Provider>
  );
}

describe('FileDropScreen', () => {
  test('ignores a stale URL fetch that resolves after a local drop selects a newer file', async () => {
    const pending = deferred<Response>();
    let fetchSignal: AbortSignal | undefined;
    globalThis.fetch = ((_input: unknown, init?: RequestInit) => {
      fetchSignal = init?.signal ?? undefined;
      return pending.promise;
    }) as typeof fetch;

    const history: string[] = [];

    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedRoot = root;
    mountedContainer = container;

    await act(async () => {
      root.render(<DropScreenHarness onSelect={(name) => history.push(name)} />);
    });

    await submitUrl(container, 'https://example.test/old.vortex');

    const dropZone = container.querySelector('.x-drop');
    expect(dropZone).not.toBeNull();
    const dropEvent = new Event('drop', { bubbles: true, cancelable: true });
    Object.defineProperty(dropEvent, 'dataTransfer', {
      value: { files: [new File(['x'], 'new.vortex')] },
    });

    await act(async () => {
      dropZone!.dispatchEvent(dropEvent);
    });

    expect(container.querySelector('[data-current]')?.textContent).toBe('new.vortex');
    expect(fetchSignal?.aborted).toBe(true);

    await act(async () => {
      pending.resolve({
        ok: true,
        blob: async () => new Blob(['y']),
      } as Response);
    });

    expect(container.querySelector('[data-current]')?.textContent).toBe('new.vortex');
    expect(history).toEqual(['new.vortex']);
  });
});
