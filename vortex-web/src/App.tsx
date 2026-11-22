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
    sessionRef.current = new ExplorerSession(() => workerRef.current);
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

  const fileCallbacks = useMemo(
    () => session.getFileCallbacks(activeGeneration),
    [session, activeGeneration],
  );

  const fileContextValue = useMemo<VortexFileContextValue | null>(
    () =>
      fileState
        ? {
            ...fileState,
            ...fileCallbacks,
          }
        : null,
    [fileState, fileCallbacks],
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
