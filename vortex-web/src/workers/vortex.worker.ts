// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// <reference lib="webworker" />

import init, { open_vortex_file, type VortexFileHandle } from '../wasm/pkg/vortex_web_wasm.js';

let initialized = false;
let fileHandle: VortexFileHandle | null = null;

self.onmessage = async (e: MessageEvent) => {
  const { type, id } = e.data;
  try {
    if (!initialized) {
      await init();
      initialized = true;
    }

    if (type === 'open') {
      // Free previous handle if any.
      if (fileHandle) {
        fileHandle.free();
        fileHandle = null;
      }
      fileHandle = await open_vortex_file(e.data.file);
      const result = {
        rowCount: Number(fileHandle.row_count),
        dtype: fileHandle.dtype,
        layoutTree: JSON.parse(fileHandle.layout_tree()),
        segments: JSON.parse(fileHandle.segment_map()),
        fileStructure: JSON.parse(fileHandle.file_structure()),
      };
      self.postMessage({ type: 'result', id, data: result });
    } else if (type === 'fetchEncodingTree') {
      if (!fileHandle) throw new Error('No file open');
      const json = await fileHandle.fetch_encoding_tree(e.data.nodeId);
      self.postMessage({ type: 'result', id, data: JSON.parse(json) });
    } else if (type === 'fetchArrayBuffer') {
      if (!fileHandle) throw new Error('No file open');
      const buf: Uint8Array = await fileHandle.fetch_array_buffer(
        e.data.layoutNodeId,
        e.data.arrayPath,
        e.data.bufferIndex,
      );
      self.postMessage({ type: 'result', id, data: buf }, [buf.buffer]);
    } else if (type === 'previewArrayData') {
      if (!fileHandle) throw new Error('No file open');
      const ipcBytes: Uint8Array = await fileHandle.preview_array_data(
        e.data.layoutNodeId,
        e.data.arrayPath,
        e.data.rowLimit,
      );
      self.postMessage({ type: 'result', id, data: ipcBytes }, [ipcBytes.buffer]);
    } else if (type === 'previewData') {
      if (!fileHandle) throw new Error('No file open');
      const ipcBytes: Uint8Array = await fileHandle.preview_data(e.data.nodeId, e.data.rowLimit);
      self.postMessage({ type: 'result', id, data: ipcBytes }, [ipcBytes.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id, error: String(err) });
  }
};
