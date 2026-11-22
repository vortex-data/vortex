// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { parseArrayNodeId } from '../swimlane/utils';

export interface BufferSelectionState {
  nodeId: string | null;
  selectedBuffer: number | null;
  bufferData: Uint8Array | null;
  loading: boolean;
  error: string | null;
}

export type FetchArrayBufferFn = (
  layoutNodeId: string,
  arrayPath: string[],
  bufferIndex: number,
) => Promise<Uint8Array>;

const INITIAL_STATE: BufferSelectionState = {
  nodeId: null,
  selectedBuffer: null,
  bufferData: null,
  loading: false,
  error: null,
};

/**
 * Coordinates async buffer reads for BuffersPane so that only the latest
 * request for the active node can update bytes, error, or loading state.
 */
export class BufferSelectionController {
  private fetchArrayBuffer: FetchArrayBufferFn;
  private requestId = 0;
  private currentNodeId: string | null = null;
  private snapshot: BufferSelectionState = INITIAL_STATE;
  private listeners = new Set<() => void>();

  constructor(fetchArrayBuffer: FetchArrayBufferFn) {
    this.fetchArrayBuffer = fetchArrayBuffer;
  }

  setFetchArrayBuffer(fetchArrayBuffer: FetchArrayBufferFn): void {
    this.fetchArrayBuffer = fetchArrayBuffer;
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): BufferSelectionState => this.snapshot;

  private setSnapshot(next: BufferSelectionState): void {
    this.snapshot = next;
    for (const listener of this.listeners) {
      listener();
    }
  }

  invalidate(): void {
    this.requestId += 1;
  }

  selectNode(nodeId: string, bufferCount: number): Promise<void> {
    this.currentNodeId = nodeId;
    if (bufferCount > 0) {
      return this.selectBuffer(0);
    }
    this.requestId += 1;
    this.setSnapshot({
      nodeId,
      selectedBuffer: null,
      bufferData: null,
      loading: false,
      error: null,
    });
    return Promise.resolve();
  }

  async selectBuffer(index: number): Promise<void> {
    const nodeId = this.currentNodeId;
    if (nodeId === null) {
      return;
    }

    const requestId = ++this.requestId;
    this.setSnapshot({
      nodeId,
      selectedBuffer: index,
      bufferData: null,
      loading: true,
      error: null,
    });

    try {
      const { layoutNodeId, arrayPath } = parseArrayNodeId(nodeId);
      const data = await this.fetchArrayBuffer(layoutNodeId, arrayPath, index);
      if (requestId !== this.requestId) {
        return;
      }
      this.setSnapshot({
        nodeId,
        selectedBuffer: index,
        bufferData: data,
        loading: false,
        error: null,
      });
    } catch (e) {
      if (requestId !== this.requestId) {
        return;
      }
      this.setSnapshot({
        nodeId,
        selectedBuffer: index,
        bufferData: null,
        loading: false,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }
}
