// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from 'react';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import type { VortexFileState, VortexFileContextValue } from './contexts/VortexFileContext';
import { VortexFileProvider } from './contexts/VortexFileContext';
import { SelectionProvider } from './contexts/SelectionContext';
import type { LayoutTreeNode } from './components/swimlane/types';
import { arrayTreeToLayoutChildren, findNodeById } from './components/swimlane/utils';
import { FileDropScreen } from './components/explorer/FileDropScreen';
import { FileHeader } from './components/explorer/FileHeader';
import { MainArea, type MainView } from './components/explorer/MainArea';
import { StatusBar } from './components/explorer/StatusBar';
import { VortexWorker } from './workers/VortexWorker';
import { fetchRemoteFile } from './remoteFile';
import { resolveDeepLink, viewForPathname } from './routes';

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [fileState, setFileState] = useState<VortexFileState | null>(null);
  const [baselineState, setBaselineState] = useState<VortexFileState | null>(null);
  const [activeSide, setActiveSide] = useState<'baseline' | 'candidate'>('candidate');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const dragCounter = useRef(0);
  const workerRef = useRef<VortexWorker | null>(null);
  const baselineWorkerRef = useRef<VortexWorker | null>(null);
  const view: MainView = viewForPathname(location.pathname);

  useEffect(() => {
    workerRef.current = new VortexWorker();
    baselineWorkerRef.current = new VortexWorker();
    return () => {
      workerRef.current?.terminate();
      baselineWorkerRef.current?.terminate();
    };
  }, []);

  const stateFromResult = useCallback(
    (file: File, result: Awaited<ReturnType<VortexWorker['openFile']>>): VortexFileState => ({
      fileName: file.name,
      fileSize: file.size,
      rowCount: result.rowCount,
      version: result.fileStructure.version,
      dtype: result.dtype,
      layoutTree: result.layoutTree,
      segments: result.segments,
      fileStructure: result.fileStructure,
    }),
    [],
  );

  const openFile = useCallback(
    async (file: File) => {
      setError(null);
      setLoading(true);
      try {
        const result = await workerRef.current!.openFile(file);
        setFileState(stateFromResult(file, result));
        setBaselineState(null);
        setActiveSide('candidate');
        navigate('/');
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setFileState(null);
      } finally {
        setLoading(false);
      }
    },
    [navigate, stateFromResult],
  );

  const openBaseline = useCallback(
    async (file: File) => {
      setError(null);
      setLoading(true);
      try {
        const result = await baselineWorkerRef.current!.openFile(file);
        setBaselineState(stateFromResult(file, result));
        setActiveSide('candidate');
        navigate('/compare');
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setBaselineState(null);
      } finally {
        setLoading(false);
      }
    },
    [navigate, stateFromResult],
  );

  const openCandidate = useCallback(
    async (file: File) => {
      setError(null);
      setLoading(true);
      try {
        const result = await workerRef.current!.openFile(file);
        setFileState(stateFromResult(file, result));
        setActiveSide('candidate');
        navigate('/compare');
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [navigate, stateFromResult],
  );

  useEffect(() => {
    const deepLink = resolveDeepLink(location.pathname, searchParams);
    if (!deepLink) return;
    if (deepLink.kind === 'error') {
      setError(deepLink.message);
      return;
    }

    if (deepLink.kind === 'file') {
      const controller = new AbortController();
      let active = true;
      async function openDeepLinkedFile(fileSource: string) {
        setError(null);
        setLoading(true);
        try {
          const file = await fetchRemoteFile(fileSource, controller.signal);
          if (!active) return;
          const result = await workerRef.current!.openFile(file);
          if (!active) return;
          setFileState(stateFromResult(file, result));
          setBaselineState(null);
          setActiveSide('candidate');
        } catch (e) {
          if (!active || (e instanceof DOMException && e.name === 'AbortError')) return;
          setError(e instanceof Error ? e.message : String(e));
          setFileState(null);
        } finally {
          if (active) setLoading(false);
        }
      }
      void openDeepLinkedFile(deepLink.source);
      return () => {
        active = false;
        controller.abort();
      };
    }

    const controller = new AbortController();
    let active = true;
    async function openDeepLink(baselineSource: string, candidateSource: string) {
      setError(null);
      setLoading(true);
      try {
        const [baselineFile, candidateFile] = await Promise.all([
          fetchRemoteFile(baselineSource, controller.signal),
          fetchRemoteFile(candidateSource, controller.signal),
        ]);
        if (!active) return;
        const [baselineResult, candidateResult] = await Promise.all([
          baselineWorkerRef.current!.openFile(baselineFile),
          workerRef.current!.openFile(candidateFile),
        ]);
        if (!active) return;
        setBaselineState(stateFromResult(baselineFile, baselineResult));
        setFileState(stateFromResult(candidateFile, candidateResult));
        setActiveSide('candidate');
      } catch (e) {
        if (!active || (e instanceof DOMException && e.name === 'AbortError')) return;
        setError(e instanceof Error ? e.message : String(e));
        setBaselineState(null);
        setFileState(null);
      } finally {
        if (active) setLoading(false);
      }
    }
    void openDeepLink(deepLink.baselineSource, deepLink.candidateSource);
    return () => {
      active = false;
      controller.abort();
    };
  }, [location.pathname, searchParams, stateFromResult]);

  const activeState =
    activeSide === 'baseline' && baselineState !== null ? baselineState : fileState;
  const activeWorkerRef = activeSide === 'baseline' ? baselineWorkerRef : workerRef;
  const setActiveState = activeSide === 'baseline' ? setBaselineState : setFileState;

  const fetchEncodingTree = useCallback(
    (nodeId: string) => activeWorkerRef.current!.fetchEncodingTree(nodeId),
    [activeWorkerRef],
  );

  const previewData = useCallback(
    (nodeId: string, rowLimit: number) => activeWorkerRef.current!.previewData(nodeId, rowLimit),
    [activeWorkerRef],
  );

  /** Clone a tree, replacing the node at targetId with a modified version. */
  const cloneTreeWithUpdate = useCallback(
    (
      root: LayoutTreeNode,
      targetId: string,
      update: (node: LayoutTreeNode) => LayoutTreeNode,
    ): LayoutTreeNode => {
      if (root.id === targetId) return update(root);
      const newChildren = root.children.map((child) =>
        cloneTreeWithUpdate(child, targetId, update),
      );
      if (newChildren === root.children) return root;
      return { ...root, children: newChildren };
    },
    [],
  );

  const expandArrayTree = useCallback(
    async (nodeId: string) => {
      // Fetch the encoding tree (may be async).
      const arrayTree = await activeWorkerRef.current!.fetchEncodingTree(nodeId);
      if (!arrayTree) return;

      setActiveState((prev) => {
        if (!prev) return prev;
        const node = findNodeById(prev.layoutTree, nodeId);
        if (!node || node.encoding !== 'vortex.flat') return prev;
        if (node.children.some((c) => c.isArrayNode)) return prev;

        const arrayChildren = arrayTreeToLayoutChildren(arrayTree, node);
        const newTree = cloneTreeWithUpdate(prev.layoutTree, nodeId, (n) => ({
          ...n,
          arrayEncodingTree: arrayTree,
          children: [...n.children, ...arrayChildren],
        }));
        return { ...prev, layoutTree: newTree };
      });
    },
    [activeWorkerRef, cloneTreeWithUpdate, setActiveState],
  );

  const fetchArrayBuffer = useCallback(
    (layoutNodeId: string, arrayPath: string[], bufferIndex: number) =>
      activeWorkerRef.current!.fetchArrayBuffer(layoutNodeId, arrayPath, bufferIndex),
    [activeWorkerRef],
  );

  const previewArrayData = useCallback(
    (layoutNodeId: string, arrayPath: string[], rowLimit: number) =>
      activeWorkerRef.current!.previewArrayData(layoutNodeId, arrayPath, rowLimit),
    [activeWorkerRef],
  );

  const fileContextValue = useMemo<VortexFileContextValue | null>(
    () =>
      activeState
        ? {
            ...activeState,
            fetchEncodingTree,
            previewData,
            expandArrayTree,
            fetchArrayBuffer,
            previewArrayData,
          }
        : null,
    [
      activeState,
      fetchEncodingTree,
      previewData,
      expandArrayTree,
      fetchArrayBuffer,
      previewArrayData,
    ],
  );

  const closeFile = useCallback(() => {
    setFileState(null);
    setBaselineState(null);
    navigate('/');
  }, [navigate]);

  const changeView = useCallback(
    (nextView: MainView) => {
      navigate(nextView === 'details' ? '/' : `/${nextView}`);
    },
    [navigate],
  );

  const viewComparisonFile = useCallback(
    (side: 'baseline' | 'candidate') => {
      setActiveSide(side);
      navigate('/');
    },
    [navigate],
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
    <VortexFileProvider value={fileContextValue!}>
      <SelectionProvider tree={fileContextValue!.layoutTree}>
        <div
          className="flex flex-col h-screen bg-vortex-white dark:bg-vortex-black relative"
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <FileHeader
            onClose={closeFile}
            view={view}
            onViewChange={changeView}
            onCompareFile={activeSide === 'baseline' ? openCandidate : openBaseline}
            comparisonName={
              activeSide === 'baseline' ? fileState?.fileName : baselineState?.fileName
            }
          />
          <MainArea
            view={view}
            baseline={baselineState ?? undefined}
            candidate={fileState ?? undefined}
            onViewComparisonFile={viewComparisonFile}
            onReplaceBaseline={openBaseline}
            onReplaceCandidate={openCandidate}
          />
          <StatusBar />
          {isDragging && (
            <div className="absolute inset-0 z-50 flex items-center justify-center bg-vortex-black/50 dark:bg-black/50 backdrop-blur-sm pointer-events-none">
              <p className="font-mono text-sm text-white/80">Drop to open file</p>
            </div>
          )}
        </div>
      </SelectionProvider>
    </VortexFileProvider>
  );
}

export default App;
