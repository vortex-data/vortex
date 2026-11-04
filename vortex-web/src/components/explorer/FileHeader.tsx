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

  return (
    <Header
      file={
        <h1 className="x-file">
          <span className="vx-mono x-file-name">{file.fileName}</span>
          <span className="vx-label-sm x-file-meta">
            v{file.version} · {formatBytes(file.fileSize)} · {formatRowCount(file.rowCount)} rows
          </span>
        </h1>
      }
      tools={<Switch label="View" value={view} options={VIEWS} onChange={onViewChange} />}
      // Below 640px the switch leaves the row, so the Menu's panel holds its twin.
      menu={VIEWS.map((o) => (
        <button
          key={o.value}
          type="button"
          className="vx-label vx-nav-link vx-header-menu-spill"
          aria-pressed={view === o.value}
          onClick={(e) => {
            onViewChange(o.value);
            e.currentTarget.closest('details')?.removeAttribute('open');
          }}
        >
          {o.label} view
        </button>
      ))}
      end={
        <button
          type="button"
          className="vx-btn vx-btn-ghost x-icon-btn"
          onClick={onClose}
          aria-label="Close file"
          title="Close file"
        >
          <Icon name="close" />
        </button>
      }
    />
  );
}
