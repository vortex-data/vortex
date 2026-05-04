import React from 'react';

export default function Header({
  sidebarOpen,
  onMenuToggle,
  categoryFilter,
  onCategoryChange,
  searchFilter,
  onSearchChange,
  viewMode,
  onViewModeChange,
  onExpandAll,
  onCollapseAll,
}) {
  return (
    <header className="sticky-header">
      <div className="header-content">
        <div className="header-left">
          <button
            className="menu-toggle"
            onClick={onMenuToggle}
            aria-label="Toggle menu"
          >
            ☰
          </button>
          <a href="/" className="logo-link">
            <picture>
                <source srcset="/Vortex_Black_NoBG.png" media="(prefers-color-scheme: light)"/>
                <source srcset="/Vortex_White_NoBG.png" media="(prefers-color-scheme: dark)"/>
                <img src="/Vortex_Black_NoBG.png" alt="Vortex" className="site-logo"/>
            </picture>
          </a>
        </div>

        <h1 className="site-title">Vortex Benchmarks</h1>

        <div className="header-center">
          <div className="filter-controls">
            <button className="control-btn" onClick={onExpandAll}>
              Expand All
            </button>
            <button className="control-btn" onClick={onCollapseAll}>
              Collapse All
            </button>
            <select
              className="category-filter"
              value={categoryFilter}
              onChange={(e) => onCategoryChange(e.target.value)}
              aria-label="Filter benchmarks by category"
            >
              <option value="all">All Benchmarks</option>
              <option value="Read/Write">Read/Write</option>
              <option value="Queries (NVMe)">Queries (NVMe)</option>
              <option value="Queries (S3)">Queries (S3)</option>
              <option value="TPC-H (SF=1)">TPC-H (SF=1)</option>
              <option value="TPC-H (SF=10)">TPC-H (SF=10)</option>
              <option value="TPC-H (SF=100)">TPC-H (SF=100)</option>
              <option value="TPC-H (SF=1000)">TPC-H (SF=1000)</option>
              <option value="TPC-DS (SF=1)">TPC-DS (SF=1)</option>
              <option value="TPC-DS (SF=10)">TPC-DS (SF=10)</option>
            </select>
            <input
              type="text"
              className="search-filter"
              placeholder="Search benchmarks..."
              value={searchFilter}
              onChange={(e) => onSearchChange(e.target.value)}
            />
          </div>
        </div>

        <div className="header-right">
          <div className="view-controls">
            <button
              className={`view-btn ${viewMode === 'grid' ? 'active' : ''}`}
              onClick={() => onViewModeChange('grid')}
              aria-label="Grid view"
            >
              ⊞
            </button>
            <button
              className={`view-btn ${viewMode === 'list' ? 'active' : ''}`}
              onClick={() => onViewModeChange('list')}
              aria-label="List view"
            >
              ☰
            </button>
          </div>
          <a
            href="https://github.com/vortex-data/vortex"
            className="repo-link"
            rel="noopener noreferrer"
            target="_blank"
          >
            <svg
              className="github-logo"
              viewBox="0 0 16 16"
              width="16"
              height="16"
              fill="currentColor"
            >
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            <span>GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
}
