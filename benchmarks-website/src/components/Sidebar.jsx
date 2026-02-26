import React from 'react';

export default function Sidebar({
  isOpen,
  groups,
  onClose,
  onGroupClick,
  onClearFilter,
  showClearFilter,
}) {
  return (
    <aside className={`sidebar ${isOpen ? 'active open' : 'collapsed'}`}>
      <nav className="sidebar-nav">
        <div className="sidebar-header">
          <h2>Navigation</h2>
          <button
            className="sidebar-close"
            onClick={onClose}
            aria-label="Close sidebar"
          >
            ×
          </button>
        </div>

        {showClearFilter && (
          <button className="clear-filter-btn" onClick={onClearFilter}>
            Clear Filter
          </button>
        )}

        <ul className="toc-list">
          {groups.map(groupName => (
            <li key={groupName}>
              <a
                href={`#group-${groupName.replace(/\s+/g, '-')}`}
                onClick={(e) => {
                  e.preventDefault();
                  onGroupClick(groupName);
                }}
              >
                {groupName}
              </a>
            </li>
          ))}
        </ul>

        <div className="sidebar-footer">
          <a
            href="https://vortex-benchmark-results-database.s3.amazonaws.com/data.json.gz"
            className="download-btn"
            download
          >
            Download Data
          </a>
        </div>
      </nav>
    </aside>
  );
}
