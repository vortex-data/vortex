// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react';
import type { VortexFileState } from './contexts/VortexFileContext';
import { VortexFileProvider } from './contexts/VortexFileContext';
import { SelectionProvider } from './contexts/SelectionContext';
import { FileDropScreen } from './components/explorer/FileDropScreen';
import { FileHeader } from './components/explorer/FileHeader';
import { MainArea } from './components/explorer/MainArea';
import { StatusBar } from './components/explorer/StatusBar';
import { VortexWorker } from './workers/VortexWorker';

function App() {
  const [fileState, setFileState] = useState<VortexFileState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const dragCounter = useRef(0);
  const workerRef = useRef<VortexWorker | null>(null);

  useEffect(() => {
    workerRef.current = new VortexWorker();
    return () => workerRef.current?.terminate();
  }, []);

  const openFile = useCallback(async (file: File) => {
    setError(null);
    setLoading(true);
    try {
      const result = await workerRef.current!.openFile(file);
      setFileState({
        fileName: file.name,
        fileSize: file.size,
        rowCount: result.rowCount,
        version: result.fileStructure.version,
        dtype: result.dtype,
        layoutTree: result.layoutTree,
        segments: result.segments,
        fileStructure: result.fileStructure,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setFileState(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const closeFile = useCallback(() => setFileState(null), []);

  const handleDragEnter = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragCounter.current++;
    if (dragCounter.current === 1) setIsDragging(true);
  }, []);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current === 0) setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      dragCounter.current = 0;
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) openFile(file);
    },
    [openFile],
  );

  if (!fileState) {
    return <FileDropScreen onFileLoaded={openFile} loading={loading} error={error} />;
  }

  return (
    <VortexFileProvider value={fileState}>
      <SelectionProvider tree={fileState.layoutTree}>
        <div
          className="flex flex-col h-screen bg-vortex-white dark:bg-vortex-black relative"
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <FileHeader onClose={closeFile} />
          <MainArea />
          <StatusBar />
          {isDragging && (
            <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm pointer-events-none">
              <p className="font-mono text-lg text-white">Drop to open file</p>
            </div>
          )}
        </div>
      </SelectionProvider>
    </VortexFileProvider>
  );
}

export default App;
