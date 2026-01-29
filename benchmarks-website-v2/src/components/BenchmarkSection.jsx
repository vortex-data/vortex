import React, { useState, useCallback, useMemo } from 'react';
import { Info, Link2 } from 'lucide-react';
import ChartContainer from './ChartContainer';
import BenchmarkSummary from './BenchmarkSummary';
import { getBenchmarkDescription, remapChartName } from '../utils';

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
}) {
  const [engineFilter, setEngineFilter] = useState('all');
  const [copiedLink, setCopiedLink] = useState(false);

  // Get unique engines from chart series
  const engines = useMemo(() => {
    const engineSet = new Set();
    charts.forEach(chart => {
      chart.series?.forEach(seriesName => {
        if (seriesName.includes(':')) {
          const engine = seriesName.split(':')[0].toLowerCase();
          engineSet.add(engine);
        }
      });
    });
    return Array.from(engineSet).sort();
  }, [charts]);

  // Filter and sort charts based on config
  const filteredCharts = useMemo(() => {
    if (!charts) return [];

    let result = charts.filter(chart => {
      // Apply keptCharts filter
      if (config.keptCharts) {
        const upperName = chart.name.toUpperCase();
        return config.keptCharts.some(kept => upperName === kept.toUpperCase());
      }
      return true;
    });

    // Sort by keptCharts order if specified
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

  // Copy link to clipboard
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

        {isExpanded && engines.length > 0 && (
          <div className="engine-filter-container">
            <span className="engine-filter-label">Filter by engine:</span>
            <button
              className={`engine-filter-btn ${engineFilter === 'all' ? 'active' : ''}`}
              onClick={() => setEngineFilter('all')}
            >
              All
            </button>
            {engines.map(engine => (
              <button
                key={engine}
                className={`engine-filter-btn ${engineFilter === engine ? 'active' : ''}`}
                onClick={() => setEngineFilter(engine)}
              >
                {engine}
              </button>
            ))}
          </div>
        )}
      </div>

      <BenchmarkSummary groupName={groupName} charts={filteredCharts} summary={summary} />

      {isExpanded && (
        <div className={`benchmark-graphs ${viewMode === 'list' ? 'list-view' : ''} ${chartCount === 1 ? 'single-chart' : ''}`}>
          {filteredCharts.map(chart => (
            <ChartContainer
              key={chart.name}
              groupName={groupName}
              chartName={chart.name}
              displayName={remapChartName(chart.name)}
              unit={config.unitOverride || chart.unit}
              config={config}
              engineFilter={engineFilter}
              onFullscreen={onFullscreen}
              commitRange={commitRange}
            />
          ))}
        </div>
      )}
    </section>
  );
}
