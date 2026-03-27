// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { FileStructureInfo, SegmentMapEntry } from '../components/swimlane/types';

/**
 * Generate a FileStructureInfo from segment entries and a total file size.
 */
export function generateFileStructure(
  segments: SegmentMapEntry[],
  fileSize: number,
): FileStructureInfo {
  const totalDataBytes = segments.reduce((sum, s) => sum + s.byteLength, 0);

  const eofSize = 8;
  const postscriptSize = 64;
  const footerSize = Math.max(256, segments.length * 16);
  const layoutSize = Math.max(128, Math.floor(fileSize * 0.02));
  const dtypeSize = Math.max(64, Math.floor(fileSize * 0.01));
  const metadataTotal = eofSize + postscriptSize + footerSize + layoutSize + dtypeSize;

  return {
    fileSize,
    version: 1,
    postscriptSize,
    totalDataBytes,
    totalMetadataBytes: metadataTotal,
  };
}
