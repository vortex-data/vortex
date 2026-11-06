// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useVortexFile } from '../../contexts/VortexFileContextCore';
import { Icon } from '../Icon';
import { Switch } from '../Switch';
import { formatBytes, formatRowCount } from '../swimlane/utils';
import { Header } from './Header';
import type { MainView } from './MainArea';

interface FileHeaderProps {
  onClose: () => void;
  view: MainView;
  onViewChange: (view: MainView) => void;
}

const VIEWS = [
  { value: 'details', label: 'Details' },
  { value: 'swimlane', label: 'Swimlane' },
] as const;

export function FileHeader({ onClose, view, onViewChange }: FileHeaderProps) {
  const file = useVortexFile();

  // Below 640px the tools leave the row, so the Menu's panel holds their twins.
  const closeMenu = (el: HTMLElement) => el.closest('details')?.removeAttribute('open');

  return (
    <Header
      tools={
        <>
          <h1 className="x-file">
            <span className="vx-mono x-file-name">{file.fileName}</span>
            <span className="vx-label-sm x-file-meta">
              v{file.version} · {formatBytes(file.fileSize)} · {formatRowCount(file.rowCount)} rows
            </span>
          </h1>
          <Switch label="View" value={view} options={VIEWS} onChange={onViewChange} />
          <button
            type="button"
            className="vx-btn vx-btn-ghost x-icon-btn"
            onClick={onClose}
            aria-label="Close file"
            title="Close file"
          >
            <Icon name="close" />
          </button>
        </>
      }
      menu={
        <>
          {VIEWS.map((o) => (
            <button
              key={o.value}
              type="button"
              className="vx-label vx-nav-link vx-header-menu-spill"
              aria-pressed={view === o.value}
              onClick={(e) => {
                onViewChange(o.value);
                closeMenu(e.currentTarget);
              }}
            >
              {o.label} view
            </button>
          ))}
          <button
            type="button"
            className="vx-label vx-nav-link vx-header-menu-spill"
            onClick={(e) => {
              closeMenu(e.currentTarget);
              onClose();
            }}
          >
            Close file
          </button>
        </>
      }
    />
  );
}
