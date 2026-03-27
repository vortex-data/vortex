// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useVortexFile } from '../../contexts/VortexFileContext';

export function FileHeader() {
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
    </div>
  );
}
