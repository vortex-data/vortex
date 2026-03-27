// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useVortexFile } from '../../contexts/VortexFileContext';

interface FileHeaderProps {
  onClose: () => void;
}

export function FileHeader({ onClose }: FileHeaderProps) {
  const file = useVortexFile();

  return (
    <div className="flex items-center gap-3 px-4 py-1.5 border-b border-vortex-grey-light dark:border-vortex-grey-dark bg-vortex-white dark:bg-vortex-black flex-shrink-0">
      <span className="font-medium text-sm text-vortex-black dark:text-vortex-white">
        {file.fileName}
      </span>
      <span
        className="text-[10px] text-vortex-grey-dark cursor-default"
        title="Vortex file format version"
      >
        v{file.version}
      </span>
      <button
        onClick={onClose}
        className="ml-auto text-xs text-vortex-grey-dark hover:text-vortex-black dark:hover:text-vortex-white transition-colors cursor-pointer"
        title="Close file"
      >
        Close
      </button>
    </div>
  );
}
