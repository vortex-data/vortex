// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterEach, describe, expect, test } from 'bun:test';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { BuffersPane } from '../src/components/detail/BuffersPane';
import type { LayoutTreeNode } from '../src/components/swimlane/types';
import type { VortexFileContextValue } from '../src/contexts/VortexFileContext';
import { VortexFileContext } from '../src/contexts/VortexFileContextCore';

class FakeNode {
  nodeType: number;
  nodeName: string;
  parentNode: FakeNode | null = null;
  childNodes: FakeNode[] = [];
  ownerDocument: FakeDocument;

  constructor(nodeType: number, nodeName: string, doc: FakeDocument) {
    this.nodeType = nodeType;
    this.nodeName = nodeName;
    this.ownerDocument = doc;
  }

  get firstChild(): FakeNode | null {
    return this.childNodes[0] ?? null;
  }

  get lastChild(): FakeNode | null {
    return this.childNodes[this.childNodes.length - 1] ?? null;
  }

  get nextSibling(): FakeNode | null {
    if (!this.parentNode) return null;
    const index = this.parentNode.childNodes.indexOf(this);
    return this.parentNode.childNodes[index + 1] ?? null;
  }

  appendChild(child: FakeNode): FakeNode {
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }

  insertBefore(child: FakeNode, ref: FakeNode | null): FakeNode {
    if (!ref) return this.appendChild(child);
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    const index = this.childNodes.indexOf(ref);
    if (index === -1) this.childNodes.push(child);
    else this.childNodes.splice(index, 0, child);
    return child;
  }

  removeChild(child: FakeNode): FakeNode {
    const index = this.childNodes.indexOf(child);
    if (index !== -1) this.childNodes.splice(index, 1);
    child.parentNode = null;
    return child;
  }

  get textContent(): string {
    return this.childNodes.map((child) => child.textContent).join('');
  }

  set textContent(value: string) {
    this.childNodes = [];
    if (value) this.appendChild(this.ownerDocument.createTextNode(value));
  }
}

class FakeText extends FakeNode {
  nodeValue: string;

  constructor(text: string, doc: FakeDocument) {
    super(3, '#text', doc);
    this.nodeValue = text;
  }

  override get textContent(): string {
    return this.nodeValue;
  }

  override set textContent(value: string) {
    this.nodeValue = String(value);
  }
}

class FakeEvent {
  type: string;
  bubbles = true;
  cancelable = true;
  defaultPrevented = false;
  target: FakeElement | null = null;
  currentTarget: FakeElement | null = null;

  constructor(type: string) {
    this.type = type;
  }

  preventDefault(): void {
    this.defaultPrevented = true;
  }

  stopPropagation(): void {}
}

class FakeElement extends FakeNode {
  tagName: string;
  namespaceURI = 'http://www.w3.org/1999/xhtml';
  attributes = new Map<string, string>();
  listeners = new Map<string, Set<(event: FakeEvent) => void>>();
  style: Record<string, string> = {};
  dataset: Record<string, string> = {};

  constructor(tag: string, doc: FakeDocument) {
    super(1, tag.toUpperCase(), doc);
    this.tagName = tag.toUpperCase();
  }

  setAttribute(key: string, value: string): void {
    this.attributes.set(key, String(value));
  }

  getAttribute(key: string): string | null {
    return this.attributes.get(key) ?? null;
  }

  removeAttribute(key: string): void {
    this.attributes.delete(key);
  }

  addEventListener(type: string, fn: (event: FakeEvent) => void): void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(fn);
  }

  removeEventListener(type: string, fn: (event: FakeEvent) => void): void {
    this.listeners.get(type)?.delete(fn);
  }

  dispatchEvent(event: FakeEvent): boolean {
    event.target = this;
    let current: FakeNode | null = this;
    while (current) {
      if (current instanceof FakeElement) {
        event.currentTarget = current;
        for (const fn of current.listeners.get(event.type) ?? []) {
          fn(event);
        }
      }
      current = current.parentNode;
    }
    return true;
  }

  querySelectorAll(tag: string): FakeElement[] {
    const out: FakeElement[] = [];
    const want = tag.toUpperCase();
    const walk = (node: FakeNode) => {
      if (node instanceof FakeElement && node.tagName === want) out.push(node);
      for (const child of node.childNodes) walk(child);
    };
    walk(this);
    return out;
  }
}

class FakeDocument extends FakeNode {
  defaultView: FakeWindow | null = null;
  documentElement: FakeElement;
  body: FakeElement;

  constructor() {
    super(9, '#document', null as unknown as FakeDocument);
    this.ownerDocument = this;
    this.documentElement = new FakeElement('html', this);
    this.body = new FakeElement('body', this);
    this.appendChild(this.documentElement);
    this.documentElement.appendChild(this.body);
  }

  createElement(tag: string): FakeElement {
    return new FakeElement(tag, this);
  }

  createTextNode(text: string): FakeText {
    return new FakeText(text, this);
  }

  createComment(): FakeNode {
    return new FakeNode(8, '#comment', this);
  }

  addEventListener(): void {}
  removeEventListener(): void {}
}

class FakeHTMLIFrameElement {}

interface FakeWindow {
  document: FakeDocument;
  navigator: { userAgent: string };
  HTMLElement: typeof FakeElement;
  HTMLIFrameElement: typeof FakeHTMLIFrameElement;
  Node: typeof FakeNode;
  Event: typeof FakeEvent;
  addEventListener(): void;
  removeEventListener(): void;
}

const fakeDoc = new FakeDocument();
const fakeWindow: FakeWindow = {
  document: fakeDoc,
  navigator: { userAgent: 'bun' },
  HTMLElement: FakeElement,
  HTMLIFrameElement: FakeHTMLIFrameElement,
  Node: FakeNode,
  Event: FakeEvent,
  addEventListener() {},
  removeEventListener() {},
};
fakeDoc.defaultView = fakeWindow;

Object.assign(globalThis, {
  window: fakeWindow,
  document: fakeDoc,
  HTMLElement: FakeElement,
  HTMLIFrameElement: FakeHTMLIFrameElement,
  Node: FakeNode,
  IS_REACT_ACT_ENVIRONMENT: true,
});

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
let mountedContainer: FakeElement | null = null;

async function mountPane(fetchArrayBuffer: VortexFileContextValue['fetchArrayBuffer']) {
  const container = fakeDoc.createElement('div');
  fakeDoc.body.appendChild(container);
  const root = createRoot(container as unknown as HTMLElement);
  mountedRoot = root;
  mountedContainer = container;

  const ctx = { fetchArrayBuffer } as unknown as VortexFileContextValue;
  await act(async () => {
    root.render(
      createElement(
        VortexFileContext.Provider,
        { value: ctx },
        createElement(BuffersPane, { node: twoBufferNode }),
      ),
    );
  });

  return { container, root };
}

afterEach(async () => {
  if (mountedRoot) {
    await act(async () => {
      mountedRoot?.unmount();
    });
    mountedRoot = null;
  }
  if (mountedContainer?.parentNode) {
    mountedContainer.parentNode.removeChild(mountedContainer);
  }
  mountedContainer = null;
});

describe('BuffersPane', () => {
  test('ignores a stale resolution when another buffer is selected on the same node', async () => {
    const reqs = new Map<number, Deferred<Uint8Array>>();
    const { container } = await mountPane((_layoutId, _path, bufferIndex) => {
      const d = deferred<Uint8Array>();
      reqs.set(bufferIndex, d);
      return d.promise;
    });

    const buttons = container.querySelectorAll('button');
    expect(buttons).toHaveLength(2);

    await act(async () => {
      buttons[1]?.dispatchEvent(new FakeEvent('click'));
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
    const { container } = await mountPane((_layoutId, _path, bufferIndex) => {
      const d = deferred<Uint8Array>();
      reqs.set(bufferIndex, d);
      return d.promise;
    });

    const buttons = container.querySelectorAll('button');
    await act(async () => {
      buttons[1]?.dispatchEvent(new FakeEvent('click'));
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

  test('ignores late resolution and rejection after unmount', async () => {
    const reqs = new Map<number, Deferred<Uint8Array>>();
    const { container, root } = await mountPane((_layoutId, _path, bufferIndex) => {
      const d = deferred<Uint8Array>();
      reqs.set(bufferIndex, d);
      return d.promise;
    });

    const buttons = container.querySelectorAll('button');
    await act(async () => {
      buttons[1]?.dispatchEvent(new FakeEvent('click'));
    });

    await act(async () => {
      root.unmount();
    });
    mountedRoot = null;

    await act(async () => {
      reqs.get(0)?.resolve(new Uint8Array([0xaa]));
      reqs.get(1)?.reject(new Error('late unmount failure'));
    });
    expect(container.textContent).toBe('');
  });
});
