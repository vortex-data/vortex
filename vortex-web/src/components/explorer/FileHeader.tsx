// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useVortexFile } from '../../contexts/VortexFileContext';
import { ThemePicker } from '../ThemePicker';

interface FileHeaderProps {
  onClose: () => void;
  viewMode?: 'raw' | 'video';
  onViewModeChange?: (mode: 'raw' | 'video') => void;
}

export function FileHeader({ onClose, viewMode, onViewModeChange }: FileHeaderProps) {
  const file = useVortexFile();

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 border-b border-vortex-grey-light/60 dark:border-white/[0.08] bg-vortex-white dark:bg-vortex-black flex-shrink-0">
      <span className="font-medium text-sm text-vortex-fg-light dark:text-vortex-fg">
        {file.fileName}
      </span>
      <span
        className="text-[10px] text-vortex-grey-dark cursor-default"
        title="Vortex file format version"
      >
        v{file.version}
      </span>
      {viewMode && onViewModeChange && (
        <div className="flex rounded-md bg-vortex-grey-lightest dark:bg-white/[0.06] p-0.5">
          {(['video', 'raw'] as const).map((mode) => (
            <button
              key={mode}
              className={`px-2 py-0.5 text-[10px] rounded-[3px] transition-colors ${
                viewMode === mode
                  ? 'bg-white dark:bg-white/[0.1] text-vortex-fg-light dark:text-vortex-fg shadow-sm'
                  : 'text-vortex-grey-dark hover:text-vortex-fg-light dark:hover:text-vortex-fg'
              }`}
              onClick={() => onViewModeChange(mode)}
            >
              {mode === 'video' ? 'Video' : 'Raw'}
            </button>
          ))}
        </div>
      )}
      <div className="ml-auto flex items-center gap-1">
        <ThemePicker />
        <button
          onClick={onClose}
          className="p-1.5 rounded-md text-vortex-grey-dark hover:text-vortex-fg-light dark:hover:text-vortex-fg hover:bg-vortex-grey-lightest dark:hover:bg-white/[0.06] transition-colors cursor-pointer"
          title="Close file"
          aria-label="Close file"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
    </div>
  );
}
