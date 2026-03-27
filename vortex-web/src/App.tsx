// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useRef, useState } from 'react';
import type { InitOutput } from './wasm/pkg/vortex_web_wasm.d.ts';
import type { VortexFileState } from './contexts/VortexFileContext';
import { VortexFileProvider } from './contexts/VortexFileContext';
import { SelectionProvider } from './contexts/SelectionContext';
import { FileDropScreen } from './components/explorer/FileDropScreen';
import { FileHeader } from './components/explorer/FileHeader';
import { MainArea } from './components/explorer/MainArea';
import { StatusBar } from './components/explorer/StatusBar';

function App() {
  const [fileState, setFileState] = useState<VortexFileState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const wasmRef = useRef<InitOutput | null>(null);

  useEffect(() => {
    import('./wasm/pkg/vortex_web_wasm.js').then(async (wasm) => {
      wasmRef.current = await wasm.default();
    });
  }, []);

  const openFile = useCallback(async (file: File) => {
    setError(null);
    setLoading(true);
    try {
      const wasm = await import('./wasm/pkg/vortex_web_wasm.js');
      if (!wasmRef.current) {
        wasmRef.current = await wasm.default();
      }
      const bytes = new Uint8Array(await file.arrayBuffer());
      const handle = wasm.open_vortex_file(bytes);

      // TODO (Phase 8): call handle.layout_tree(), handle.segment_map(), handle.file_structure()
      // to populate a full VortexFileState and call setFileState(...)
      void handle.row_count;
      void handle.dtype;
      handle.free();

      setError('WASM integration pending — use Storybook for now');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setFileState(null);
    } finally {
      setLoading(false);
    }
  }, []);

  if (!fileState) {
    return <FileDropScreen onFileLoaded={openFile} loading={loading} error={error} />;
  }

  return (
    <VortexFileProvider value={fileState}>
      <SelectionProvider tree={fileState.layoutTree}>
        <div className="flex flex-col h-screen bg-vortex-white dark:bg-vortex-black">
          <FileHeader />
          <MainArea />
          <StatusBar />
        </div>
      </SelectionProvider>
    </VortexFileProvider>
  );
}

export default App;
