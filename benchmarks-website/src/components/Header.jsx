import React, { useMemo } from 'react';
import {
  FILTER_DIMENSIONS,
  collectFilterValues,
  GROUP_FILTER_DIMENSIONS,
  collectGroupFilterValues,
  WORKLOAD_LABELS,
} from '../lib/config';

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
  globalFilters,
  onGlobalFilterChange,
  hasActiveGlobalFilters,
  onClearGlobalFilters,
  allSeriesNames,
  groupFilters,
  onGroupFilterChange,
  hasActiveGroupFilters,
  onClearGroupFilters,
  allGroupNames,
}) {
  // Compute available options for each global filter dimension
  const filterOptions = useMemo(() => {
    if (!allSeriesNames?.length) return {};
    return collectFilterValues(allSeriesNames);
  }, [allSeriesNames]);

  // Compute available options for each group filter dimension
  const groupFilterOptions = useMemo(() => {
    if (!allGroupNames?.length) return {};
    return collectGroupFilterValues(allGroupNames);
  }, [allGroupNames]);

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
            <img
              src="/vortex_black_nobg.svg"
              alt="Vortex"
              className="site-logo"
            />
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

      {/* Group-level filter bar — workload, storage, scale factor */}
      {allGroupNames?.length > 0 && (
        <div className="global-filter-bar">
          {GROUP_FILTER_DIMENSIONS.map(dim => {
            const options = groupFilterOptions[dim.key] || [];
            if (options.length <= 1) return null;
            return (
              <div key={dim.key} className="global-filter-group">
                <span className="global-filter-dim-label">{dim.label}</span>
                <div className="global-filter-buttons">
                  <button
                    className={`global-filter-btn ${groupFilters?.[dim.key] === 'all' ? 'active' : ''}`}
                    onClick={() => onGroupFilterChange(dim.key, 'all')}
                  >
                    All
                  </button>
                  {options.map(val => (
                    <button
                      key={val}
                      className={`global-filter-btn ${groupFilters?.[dim.key] === val ? 'active' : ''}`}
                      onClick={() => onGroupFilterChange(dim.key, val)}
                    >
                      {WORKLOAD_LABELS[val] || val}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
          {/* Series-level filters (Engine, Format) inline */}
          {globalFilters && allSeriesNames?.length > 0 && FILTER_DIMENSIONS.map(dim => {
            const options = filterOptions[dim.key] || [];
            if (options.length <= 1) return null;
            return (
              <div key={dim.key} className="global-filter-group">
                <span className="global-filter-dim-label">{dim.label}</span>
                <div className="global-filter-buttons">
                  <button
                    className={`global-filter-btn ${globalFilters[dim.key] === 'all' ? 'active' : ''}`}
                    onClick={() => onGlobalFilterChange(dim.key, 'all')}
                  >
                    All
                  </button>
                  {options.map(val => (
                    <button
                      key={val}
                      className={`global-filter-btn ${globalFilters[dim.key] === val ? 'active' : ''}`}
                      onClick={() => onGlobalFilterChange(dim.key, val)}
                    >
                      {val}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
          {(hasActiveGroupFilters || hasActiveGlobalFilters) && (
            <button className="global-filter-clear" onClick={() => {
              onClearGroupFilters();
              onClearGlobalFilters();
            }}>
              Clear All
            </button>
          )}
        </div>
      )}
    </header>
  );
}
