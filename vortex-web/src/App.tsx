// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type DragEvent,
} from 'react';
import type { VortexFileContextValue } from './contexts/VortexFileContext';
import { VortexFileProvider } from './contexts/VortexFileContext';
import { SelectionProvider } from './contexts/SelectionContext';
import { FileDropScreen } from './components/explorer/FileDropScreen';
import { FileHeader } from './components/explorer/FileHeader';
import { MainArea, type MainView } from './components/explorer/MainArea';
import { StatusBar } from './components/explorer/StatusBar';
import { ExplorerSession } from './explorerSession';
import { VortexWorker } from './workers/VortexWorker';

function App() {
  const [isDragging, setIsDragging] = useState(false);
  const [view, setView] = useState<MainView>('details');
  const dragCounter = useRef(0);
  const workerRef = useRef<VortexWorker | null>(null);
  const sessionRef = useRef<ExplorerSession | null>(null);

  if (!sessionRef.current) {
    sessionRef.current = new ExplorerSession({
      openFile: (file, generation) => workerRef.current!.openFile(file, generation),
      closeFile: (generation) => workerRef.current?.closeFile(generation),
      fetchEncodingTree: (nodeId, generation) =>
        workerRef.current!.fetchEncodingTree(nodeId, generation),
      fetchArrayBuffer: (layoutNodeId, arrayPath, bufferIndex, generation) =>
        workerRef.current!.fetchArrayBuffer(layoutNodeId, arrayPath, bufferIndex, generation),
      previewArrayData: (layoutNodeId, arrayPath, rowLimit, generation) =>
        workerRef.current!.previewArrayData(layoutNodeId, arrayPath, rowLimit, generation),
      previewData: (nodeId, rowLimit, generation) =>
        workerRef.current!.previewData(nodeId, rowLimit, generation),
    });
  }
  const session = sessionRef.current;

  useEffect(() => {
    const worker = new VortexWorker();
    workerRef.current = worker;
    return () => {
      worker.terminate();
      if (workerRef.current === worker) {
        workerRef.current = null;
      }
    };
  }, []);

  const { fileState, activeGeneration, loading, error } = useSyncExternalStore(
    session.subscribe,
    session.getSnapshot,
  );

  const openFile = useCallback((file: File) => session.openFile(file), [session]);

  const closeFile = useCallback(() => session.closeFile(), [session]);

  const fileContextValue = useMemo<VortexFileContextValue | null>(
    () =>
      fileState
        ? {
            ...fileState,
            fetchEncodingTree: (nodeId: string) =>
              session.fetchEncodingTree(nodeId, activeGeneration),
            previewData: (nodeId: string, rowLimit: number) =>
              session.previewData(nodeId, rowLimit, activeGeneration),
            expandArrayTree: (nodeId: string) => session.expandArrayTree(nodeId, activeGeneration),
            fetchArrayBuffer: (layoutNodeId: string, arrayPath: string[], bufferIndex: number) =>
              session.fetchArrayBuffer(layoutNodeId, arrayPath, bufferIndex, activeGeneration),
            previewArrayData: (layoutNodeId: string, arrayPath: string[], rowLimit: number) =>
              session.previewArrayData(layoutNodeId, arrayPath, rowLimit, activeGeneration),
          }
        : null,
    [fileState, activeGeneration, session],
  );

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

  if (!fileContextValue) {
    return <FileDropScreen onFileLoaded={openFile} loading={loading} error={error} />;
  }

  return (
    <VortexFileProvider value={fileContextValue}>
      <SelectionProvider key={activeGeneration} tree={fileContextValue.layoutTree}>
        {/* The whole window takes a dropped file, so a new file can replace the open one. */}
        {/* biome-ignore lint/a11y/noStaticElementInteractions: drag and drop has no keyboard form; Close and the open screen's button do. */}
        <div
          className="x-app"
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <a className="vx-skip-link" href="#main">
            Skip to content
          </a>
          <FileHeader onClose={closeFile} view={view} onViewChange={setView} />
          <MainArea view={view} />
          <StatusBar />
          {isDragging && (
            <div className="x-drop-veil">
              <p className="vx-mono">Drop to open the file</p>
            </div>
          )}
        </div>
      </SelectionProvider>
    </VortexFileProvider>
  );
}

export default App;
