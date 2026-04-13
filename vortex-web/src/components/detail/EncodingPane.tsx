// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useEffect, useState } from 'react';
import { useVortexFile } from '../../contexts/VortexFileContext';
import type { LayoutTreeNode, ArrayEncodingNode } from '../swimlane/types';
import { shortEncoding, formatBytes } from '../swimlane/utils';

interface EncodingPaneProps {
  node: LayoutTreeNode;
}

/**
 * Displays the array encoding tree inside a flat layout.
 * If the tree is not inlined in the layout metadata, fetches it
 * asynchronously from the segment data via the worker.
 */
export function EncodingPane({ node }: EncodingPaneProps) {
  const { fetchEncodingTree } = useVortexFile();
  const [fetchedTree, setFetchedTree] = useState<ArrayEncodingNode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tree = node.arrayEncodingTree ?? fetchedTree;

  useEffect(() => {
    // If already inlined, nothing to fetch.
    if (node.arrayEncodingTree) return;

    // Need at least one segment to fetch from.
    if (node.segmentIds.length === 0) return;

    let cancelled = false;
    setLoading(true);
    setError(null);
    setFetchedTree(null);

    fetchEncodingTree(node.id)
      .then((result) => {
        if (!cancelled) setFetchedTree(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [node.id, node.arrayEncodingTree, node.segmentIds, fetchEncodingTree]);

  if (loading) {
    return <div className="text-xs text-vortex-grey-dark">Loading encoding tree…</div>;
  }

  if (error) {
    return <div className="text-xs text-red-500">Error: {error}</div>;
  }

  if (!tree) {
    return (
      <div className="text-xs text-vortex-grey-dark">
        No array encoding tree available for this node.
      </div>
    );
  }

  return (
    <div className="text-xs">
      <pre className="font-mono text-[10px] text-vortex-grey-dark bg-vortex-grey-lightest dark:bg-white/[0.03] rounded p-2 overflow-auto">
        {renderArrayTree(tree, 0)}
      </pre>
    </div>
  );
}

function renderArrayTree(node: ArrayEncodingNode, indent: number): string {
  const prefix = '  '.repeat(indent);
  const enc = shortEncoding(node.encoding);
  const parts = [enc];
  if (node.bufferLengths.length > 0) {
    const bufs = node.bufferLengths.map((b) => formatBytes(b)).join(', ');
    parts.push(`buffers: [${bufs}]`);
  }
  if (node.metadataBytes > 0) {
    parts.push(`meta: ${formatBytes(node.metadataBytes)}`);
  }
  const line = `${prefix}${parts.join('  ')}`;

  if (node.children.length === 0) return line;

  const childLines = node.children.map((child) => renderArrayTree(child, indent + 1));
  return [line, ...childLines].join('\n');
}
