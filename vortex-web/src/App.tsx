// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from 'react';
import type { VortexFileState, VortexFileContextValue } from './contexts/VortexFileContext';
import { VortexFileProvider } from './contexts/VortexFileContext';
import { SelectionProvider } from './contexts/SelectionContext';
import type { LayoutTreeNode } from './components/swimlane/types';
import { arrayTreeToLayoutChildren, findNodeById } from './components/swimlane/utils';
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

  const fetchEncodingTree = useCallback(
    (nodeId: string) => workerRef.current!.fetchEncodingTree(nodeId),
    [],
  );

  const previewData = useCallback(
    (nodeId: string, rowLimit: number) => workerRef.current!.previewData(nodeId, rowLimit),
    [],
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
      const arrayTree = await workerRef.current!.fetchEncodingTree(nodeId);
      if (!arrayTree) return;

      setFileState((prev) => {
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
    [cloneTreeWithUpdate],
  );

  const fetchArrayBuffer = useCallback(
    (layoutNodeId: string, arrayPath: string[], bufferIndex: number) =>
      workerRef.current!.fetchArrayBuffer(layoutNodeId, arrayPath, bufferIndex),
    [],
  );

  const previewArrayData = useCallback(
    (layoutNodeId: string, arrayPath: string[], rowLimit: number) =>
      workerRef.current!.previewArrayData(layoutNodeId, arrayPath, rowLimit),
    [],
  );

  const fileContextValue = useMemo<VortexFileContextValue | null>(
    () =>
      fileState
        ? {
            ...fileState,
            fetchEncodingTree,
            previewData,
            expandArrayTree,
            fetchArrayBuffer,
            previewArrayData,
          }
        : null,
    [
      fileState,
      fetchEncodingTree,
      previewData,
      expandArrayTree,
      fetchArrayBuffer,
      previewArrayData,
    ],
  );

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
          <FileHeader onClose={closeFile} />
          <MainArea />
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
