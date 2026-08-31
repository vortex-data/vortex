// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useVortexFile } from '../../contexts/VortexFileContextCore';
import { ThemePicker } from '../ThemePicker';
import type { MainView } from './MainArea';

interface FileHeaderProps {
  onClose: () => void;
  view: MainView;
  onViewChange: (view: MainView) => void;
  onCompareFile?: (file: File) => void;
  comparisonName?: string;
}

export function FileHeader({
  onClose,
  view,
  onViewChange,
  onCompareFile,
  comparisonName,
}: FileHeaderProps) {
  const file = useVortexFile();
  const views: MainView[] = comparisonName
    ? ['details', 'swimlane', 'compare']
    : ['details', 'swimlane'];

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
      <div className="ml-auto flex items-center gap-2">
        {/* Primary view switch — sits with the global controls in the header. */}
        <div className="flex rounded-md bg-vortex-grey-lightest dark:bg-white/[0.06] p-0.5">
          {views.map((v) => (
            <button
              key={v}
              className={`px-3 py-0.5 text-[11px] rounded-[3px] transition-colors ${
                view === v
                  ? 'bg-white dark:bg-white/[0.1] text-vortex-fg-light dark:text-vortex-fg shadow-sm font-medium'
                  : 'text-vortex-grey-dark hover:text-vortex-fg-light dark:hover:text-vortex-fg'
              }`}
              onClick={() => onViewChange(v)}
            >
              {v === 'details' ? 'Details' : v === 'swimlane' ? 'Swimlane' : 'Compare'}
            </button>
          ))}
        </div>

        <label className="cursor-pointer rounded-md px-2 py-1 text-[11px] text-vortex-grey-dark hover:bg-vortex-grey-lightest dark:hover:bg-white/[0.06]">
          {comparisonName ? `Compare with: ${comparisonName}` : 'Compare…'}
          <input
            className="hidden"
            type="file"
            accept=".vortex,.vtx"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onCompareFile?.(file);
              event.target.value = '';
            }}
          />
        </label>

        <div className="flex items-center gap-1">
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
    </div>
  );
}
