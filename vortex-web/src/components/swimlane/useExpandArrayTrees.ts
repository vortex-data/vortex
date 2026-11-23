// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useEffect, useRef } from 'react';
import type { VortexFileContextValue } from '../../contexts/VortexFileContextCore';
import { findNodeById, isFlatLayout } from './utils';

type RequestStatus = 'pending' | 'failed' | 'done';

/**
 * Lazily expands flat layout nodes that are currently open in a tree view.
 * Pending requests are deduplicated across toggles; failed requests are caught
 * and only retried after the node is collapsed and explicitly expanded again.
 */
export function useExpandArrayTrees(expanded: Set<string>, file: VortexFileContextValue) {
  const requestsRef = useRef(new Map<string, RequestStatus>());
  const expandedRef = useRef(expanded);
  expandedRef.current = expanded;

  useEffect(() => {
    for (const [id, status] of requestsRef.current) {
      if (status === 'failed' && !expanded.has(id)) {
        requestsRef.current.delete(id);
      }
    }

    for (const id of expanded) {
      if (requestsRef.current.has(id)) continue;
      const node = findNodeById(file.layoutTree, id);
      if (node && isFlatLayout(node) && !node.children.some((c) => c.isArrayNode)) {
        requestsRef.current.set(id, 'pending');
        Promise.resolve(file.expandArrayTree(id)).then(
          () => {
            requestsRef.current.set(id, 'done');
          },
          (err) => {
            console.error('Explorer: failed to expand array tree', err);
            if (expandedRef.current.has(id)) {
              requestsRef.current.set(id, 'failed');
            } else {
              requestsRef.current.delete(id);
            }
          },
        );
      }
    }
  }, [expanded, file]);
}
