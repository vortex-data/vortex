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
import { VideoExplorer } from './components/video/VideoExplorer';
import { VortexWorker } from './workers/VortexWorker';

type ExplorerViewMode = 'raw' | 'video';

interface OpenFileBundle {
  vortexFile: File;
  mediaFile?: File;
}

function normalizeExtension(name: string): string {
  const lastDot = name.lastIndexOf('.');
  return lastDot >= 0 ? name.slice(lastDot).toLowerCase() : '';
}

function stripExtension(name: string): string {
  const lastDot = name.lastIndexOf('.');
  return (lastDot >= 0 ? name.slice(0, lastDot) : name).toLowerCase();
}

function isVortexFile(file: File): boolean {
  const ext = normalizeExtension(file.name);
  return ext === '.vortex' || ext === '.vtx';
}

function isMp4File(file: File): boolean {
  return normalizeExtension(file.name) === '.mp4';
}

function resolveOpenFileBundle(files: File[]): OpenFileBundle {
  const vortexFiles = files.filter(isVortexFile);
  if (vortexFiles.length === 0) {
    throw new Error('Drop at least one .vortex file.');
  }

  const mp4ByBase = new Map<string, File>();
  for (const file of files.filter(isMp4File)) {
    mp4ByBase.set(stripExtension(file.name), file);
  }

  const matchedVortex = vortexFiles.find((file) => mp4ByBase.has(stripExtension(file.name)));
  const vortexFile = matchedVortex ?? vortexFiles[0];
  const mediaFile = mp4ByBase.get(stripExtension(vortexFile.name));

  return { vortexFile, mediaFile };
}

function App() {
  const [fileState, setFileState] = useState<VortexFileState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [viewMode, setViewMode] = useState<ExplorerViewMode>('raw');
  const dragCounter = useRef(0);
  const workerRef = useRef<VortexWorker | null>(null);
  const mediaUrlRef = useRef<string | null>(null);

  const revokePairedMedia = useCallback(() => {
    if (mediaUrlRef.current) {
      URL.revokeObjectURL(mediaUrlRef.current);
      mediaUrlRef.current = null;
    }
  }, []);

  useEffect(() => {
    workerRef.current = new VortexWorker();
    return () => {
      workerRef.current?.terminate();
      revokePairedMedia();
    };
  }, [revokePairedMedia]);

  const openFileBundle = useCallback(
    async ({ vortexFile, mediaFile }: OpenFileBundle) => {
      setError(null);
      setLoading(true);
      try {
        const result = await workerRef.current!.openFile(vortexFile);
        revokePairedMedia();

        const pairedMedia =
          result.kind === 'videoIndex'
            ? mediaFile
              ? (() => {
                  const objectUrl = URL.createObjectURL(mediaFile);
                  mediaUrlRef.current = objectUrl;
                  return { fileName: mediaFile.name, objectUrl, hasMedia: true as const };
                })()
              : { fileName: null, objectUrl: null, hasMedia: false as const }
            : undefined;

        setFileState({
          kind: result.kind,
          fileName: vortexFile.name,
          fileSize: vortexFile.size,
          rowCount: result.rowCount,
          version: result.fileStructure.version,
          dtype: result.dtype,
          layoutTree: result.layoutTree,
          segments: result.segments,
          fileStructure: result.fileStructure,
          videoIndex: result.videoIndex,
          pairedMedia,
        });
        setViewMode(result.kind === 'videoIndex' ? 'video' : 'raw');
      } catch (e) {
        revokePairedMedia();
        setError(e instanceof Error ? e.message : String(e));
        setFileState(null);
        setViewMode('raw');
      } finally {
        setLoading(false);
      }
    },
    [revokePairedMedia],
  );

  const openLocalFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      try {
        const bundle = resolveOpenFileBundle(files);
        await openFileBundle(bundle);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [openFileBundle],
  );

  const attachMediaFile = useCallback(
    (file: File) => {
      if (!isMp4File(file)) {
        setError('Attach an .mp4 file for video playback.');
        return;
      }

      revokePairedMedia();
      const objectUrl = URL.createObjectURL(file);
      mediaUrlRef.current = objectUrl;
      setFileState((prev) =>
        prev?.kind === 'videoIndex'
          ? {
              ...prev,
              pairedMedia: {
                fileName: file.name,
                objectUrl,
                hasMedia: true,
              },
            }
          : prev,
      );
    },
    [revokePairedMedia],
  );

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

  const closeFile = useCallback(() => {
    revokePairedMedia();
    setFileState(null);
    setViewMode('raw');
    setError(null);
  }, [revokePairedMedia]);

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
      const files = Array.from(e.dataTransfer.files);
      void openLocalFiles(files);
    },
    [openLocalFiles],
  );

  if (!fileContextValue) {
    return <FileDropScreen onFilesLoaded={openLocalFiles} loading={loading} error={error} />;
  }

  const canShowVideoMode = fileContextValue.kind === 'videoIndex';
  const activeViewMode = canShowVideoMode ? viewMode : 'raw';

  return (
    <VortexFileProvider value={fileContextValue}>
      <SelectionProvider tree={fileContextValue.layoutTree}>
        <div
          className="flex flex-col h-screen bg-vortex-white dark:bg-vortex-black relative"
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <FileHeader
            onClose={closeFile}
            viewMode={canShowVideoMode ? activeViewMode : undefined}
            onViewModeChange={canShowVideoMode ? setViewMode : undefined}
          />
          {activeViewMode === 'video' && fileContextValue.videoIndex ? (
            <VideoExplorer onAttachMedia={attachMediaFile} />
          ) : (
            <>
              <MainArea />
              <StatusBar />
            </>
          )}
          {isDragging && (
            <div className="absolute inset-0 z-50 flex items-center justify-center bg-vortex-black/50 dark:bg-black/50 backdrop-blur-sm pointer-events-none">
              <p className="font-mono text-sm text-white/80">Drop files to open</p>
            </div>
          )}
        </div>
      </SelectionProvider>
    </VortexFileProvider>
  );
}

export default App;
