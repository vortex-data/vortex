// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// <reference lib="webworker" />

import init, { open_vortex_file, type VortexFileHandle } from '../wasm/pkg/vortex_web_wasm.js';

interface ActiveHandle {
  handle: VortexFileHandle;
  inFlight: number;
  retired: boolean;
}

let initPromise: Promise<unknown> | null = null;
let openSeq = 0;
let activeHandle: ActiveHandle | null = null;

function ensureInit(): Promise<unknown> {
  if (!initPromise) {
    initPromise = init().catch((err) => {
      initPromise = null;
      throw err;
    });
  }
  return initPromise;
}

function retireCurrent(): void {
  const prev = activeHandle;
  if (!prev) return;
  activeHandle = null;
  prev.retired = true;
  if (prev.inFlight === 0) {
    prev.handle.free();
  }
}

async function withActiveHandle<T>(fn: (handle: VortexFileHandle) => Promise<T>): Promise<T> {
  const entry = activeHandle;
  if (!entry) throw new Error('No file open');
  entry.inFlight++;
  try {
    return await fn(entry.handle);
  } finally {
    entry.inFlight--;
    if (entry.retired && entry.inFlight === 0) {
      entry.handle.free();
    }
  }
}

self.onmessage = async (e: MessageEvent) => {
  const { type, id } = e.data;
  try {
    if (type === 'close') {
      openSeq++;
      retireCurrent();
      return;
    }

    if (type === 'open') {
      const seq = ++openSeq;
      await ensureInit();
      const handle = await open_vortex_file(e.data.file);
      if (seq !== openSeq) {
        handle.free();
        self.postMessage({ type: 'error', id, error: 'Stale file open' });
        return;
      }
      let result: {
        rowCount: number;
        dtype: string;
        layoutTree: unknown;
        segments: unknown;
        fileStructure: unknown;
      };
      try {
        result = {
          rowCount: Number(handle.row_count),
          dtype: handle.dtype,
          layoutTree: JSON.parse(handle.layout_tree()),
          segments: JSON.parse(handle.segment_map()),
          fileStructure: JSON.parse(handle.file_structure()),
        };
      } catch (err) {
        handle.free();
        throw err;
      }
      retireCurrent();
      activeHandle = { handle, inFlight: 0, retired: false };
      self.postMessage({ type: 'result', id, data: result });
    } else if (type === 'fetchEncodingTree') {
      const json = await withActiveHandle((h) => h.fetch_encoding_tree(e.data.nodeId));
      self.postMessage({ type: 'result', id, data: JSON.parse(json) });
    } else if (type === 'fetchArrayBuffer') {
      const buf = await withActiveHandle((h) =>
        h.fetch_array_buffer(e.data.layoutNodeId, e.data.arrayPath, e.data.bufferIndex),
      );
      self.postMessage({ type: 'result', id, data: buf }, [buf.buffer]);
    } else if (type === 'previewArrayData') {
      const ipcBytes = await withActiveHandle((h) =>
        h.preview_array_data(e.data.layoutNodeId, e.data.arrayPath, e.data.rowLimit),
      );
      self.postMessage({ type: 'result', id, data: ipcBytes }, [ipcBytes.buffer]);
    } else if (type === 'previewData') {
      const ipcBytes = await withActiveHandle((h) =>
        h.preview_data(e.data.nodeId, e.data.rowLimit),
      );
      self.postMessage({ type: 'result', id, data: ipcBytes }, [ipcBytes.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id, error: String(err) });
  }
};
