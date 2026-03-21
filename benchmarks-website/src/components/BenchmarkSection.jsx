import React, { useState, useCallback, useMemo } from 'react';
import { Info, Link2 } from 'lucide-react';
import ChartContainer from './ChartContainer';
import LazyChart from './LazyChart';
import BenchmarkSummary from './BenchmarkSummary';
import { getBenchmarkDescription, remapChartName } from '../lib/utils';
import {
  FILTER_DIMENSIONS,
  collectFilterValues,
  defaultFilters,
} from '../lib/config';

export default function BenchmarkSection({
  groupName,
  charts,
  config,
  isExpanded,
  onToggle,
  viewMode,
  onFullscreen,
  commitRange,
  summary,
  globalFilters,
}) {
  // Per-section local filter overrides (same shape as globalFilters)
  const [localFilters, setLocalFilters] = useState(defaultFilters);
  const [copiedLink, setCopiedLink] = useState(false);

  // Collect all series names in this section for the local filter bar
  const sectionSeriesNames = useMemo(() => {
    const names = new Set();
    charts.forEach(chart => {
      chart.series?.forEach(name => names.add(name));
    });
    return [...names];
  }, [charts]);

  // Compute available filter values for this section
  const sectionFilterValues = useMemo(() => {
    return collectFilterValues(sectionSeriesNames);
  }, [sectionSeriesNames]);

  // Merge global + local filters: local overrides global when not 'all'
  const mergedFilters = useMemo(() => {
    const merged = {};
    for (const dim of FILTER_DIMENSIONS) {
      const local = localFilters[dim.key];
      const global = globalFilters?.[dim.key];
      // Local takes precedence if set; otherwise use global
      merged[dim.key] = (local && local !== 'all') ? local : (global || 'all');
    }
    return merged;
  }, [localFilters, globalFilters]);

  const hasLocalFilters = useMemo(() => {
    return Object.values(localFilters).some(v => v !== 'all');
  }, [localFilters]);

  const handleLocalFilterChange = useCallback((key, value) => {
    setLocalFilters(prev => ({ ...prev, [key]: value }));
  }, []);

  const clearLocalFilters = useCallback(() => {
    setLocalFilters(defaultFilters());
  }, []);

  // Filter and sort charts based on config
  const filteredCharts = useMemo(() => {
    if (!charts) return [];

    let result = charts.filter(chart => {
      if (config.keptCharts) {
        const upperName = chart.name.toUpperCase();
        return config.keptCharts.some(kept => upperName === kept.toUpperCase());
      }
      return true;
    });

    if (config.keptCharts) {
      const orderMap = new Map(config.keptCharts.map((name, idx) => [name.toUpperCase(), idx]));
      result.sort((a, b) => {
        const aIdx = orderMap.get(a.name.toUpperCase()) ?? 999;
        const bIdx = orderMap.get(b.name.toUpperCase()) ?? 999;
        return aIdx - bIdx;
      });
    }

    return result;
  }, [charts, config]);

  const handleCopyLink = useCallback((e) => {
    e.stopPropagation();
    const url = `${window.location.origin}${window.location.pathname}#group-${groupName.replace(/\s+/g, '-')}`;
    navigator.clipboard.writeText(url);
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 2000);
  }, [groupName]);

  const description = getBenchmarkDescription(groupName);
  const hasData = filteredCharts.length > 0;
  const chartCount = filteredCharts.length;

  // Determine which filter dimensions have meaningful options in this section
  const activeDimensions = FILTER_DIMENSIONS.filter(
    dim => (sectionFilterValues[dim.key] || []).length > 1
  );

  return (
    <section
      id={`group-${groupName.replace(/\s+/g, '-')}`}
      className={`benchmark-set ${isExpanded ? 'expanded' : ''} ${hasData ? '' : 'no-data'}`}
    >
      <div className="sticky-header-container">
        <div className="benchmark-header" onClick={hasData ? onToggle : undefined}>
          <span className="collapse-icon">{isExpanded ? '▼' : '▶'}</span>
          <div className="title-wrapper">
            <h2 className="benchmark-title">
              {groupName}
              <button
                className={`group-link-btn ${copiedLink ? 'copied' : ''}`}
                onClick={handleCopyLink}
                data-tooltip={copiedLink ? 'Copied!' : 'Copy link'}
              >
                <Link2 size={14} />
              </button>
            </h2>
            {description && (
              <span className="info-icon" data-tooltip={description}>
                <Info size={12} />
              </span>
            )}
            <div className="benchmark-meta">
              <span>{chartCount} {chartCount === 1 ? 'CHART' : 'CHARTS'}</span>
            </div>
          </div>
        </div>

        {isExpanded && activeDimensions.length > 0 && (
          <div className="engine-filter-container">
            {activeDimensions.map(dim => (
              <div key={dim.key} className="section-filter-group">
                <span className="engine-filter-label">{dim.label}:</span>
                <button
                  className={`engine-filter-btn ${localFilters[dim.key] === 'all' ? 'active' : ''}`}
                  onClick={() => handleLocalFilterChange(dim.key, 'all')}
                >
                  All
                </button>
                {sectionFilterValues[dim.key].map(val => (
                  <button
                    key={val}
                    className={`engine-filter-btn ${localFilters[dim.key] === val ? 'active' : ''}`}
                    onClick={() => handleLocalFilterChange(dim.key, val)}
                  >
                    {val}
                  </button>
                ))}
              </div>
            ))}
            {hasLocalFilters && (
              <button className="engine-filter-btn clear-local" onClick={clearLocalFilters}>
                Clear
              </button>
            )}
          </div>
        )}
      </div>

      <BenchmarkSummary groupName={groupName} charts={filteredCharts} summary={summary} />

      {isExpanded && (
        <div className={`benchmark-graphs ${viewMode === 'list' ? 'list-view' : ''} ${chartCount === 1 ? 'single-chart' : ''}`}>
          {filteredCharts.map(chart => (
            <LazyChart key={chart.name}>
              <ChartContainer
                groupName={groupName}
                chartName={chart.name}
                displayName={remapChartName(chart.name)}
                unit={config.unitOverride || chart.unit}
                config={config}
                filters={mergedFilters}
                onFullscreen={onFullscreen}
                commitRange={commitRange}
              />
            </LazyChart>
          ))}
        </div>
      )}
    </section>
  );
}
