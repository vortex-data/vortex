import React from 'react';
import { formatTime } from '../utils';

// BenchmarkSummary now uses pre-computed summary from metadata (passed via props)
// instead of fetching all chart data
export default function BenchmarkSummary({ groupName, charts, summary }) {
  // Use pre-computed summary from metadata
  const summaryData = summary;

  if (!summaryData) return null;

  // Query benchmarks (Clickbench, TPC-H, TPC-DS, etc.)
  if (summaryData.type === 'queryBenchmark' && summaryData.rankings?.length > 0) {
    return (
      <div className="benchmark-scores-summary">
        <h3 className="scores-title">{summaryData.title || 'Performance Summary'}</h3>
        <div className="scores-list">
          {summaryData.rankings.map((item, idx) => (
            <div key={item.name} className="score-item">
              <span className="score-rank">#{idx + 1}</span>
              <span className="score-series">{item.name}</span>
              <span className="score-metrics">
                <span className="score-value">{item.score.toFixed(2)}x</span>
                <span className="score-runtime">{formatTime(item.totalRuntime)}</span>
              </span>
            </div>
          ))}
        </div>
        <div className="scores-explanation">
          {summaryData.explanation || 'Score: geometric mean of query time ratio to fastest (lower is better)'}
        </div>
      </div>
    );
  }

  if (summaryData.type === 'randomAccess' && summaryData.rankings?.length > 0) {
    return (
      <div className="benchmark-scores-summary">
        <h3 className="scores-title">{summaryData.title || 'Random Access Performance'}</h3>
        <div className="scores-list">
          {summaryData.rankings.map((item, idx) => (
            <div key={item.name} className="score-item">
              <span className="score-rank">#{idx + 1}</span>
              <span className="score-series">{item.name}</span>
              <span className="score-metrics">
                <span className="score-value">{formatTime(item.time)}</span>
                <span className="score-runtime">{item.ratio.toFixed(2)}x</span>
              </span>
            </div>
          ))}
        </div>
        <div className="scores-explanation">
          {summaryData.explanation || 'Random access time | Ratio to fastest (lower is better)'}
        </div>
      </div>
    );
  }

  if (summaryData.type === 'compression') {
    return (
      <div className="benchmark-scores-summary">
        <h3 className="scores-title">{summaryData.title || 'Compression Throughput vs Parquet'}</h3>
        <div className="scores-list">
          {summaryData.compressRatio && (
            <div className="score-item">
              <span className="score-rank">⚡</span>
              <span className="score-series">Write Speed (Compression)</span>
              <span className="score-metrics">
                <span className="score-value">{summaryData.compressRatio.toFixed(2)}x</span>
              </span>
            </div>
          )}
          {summaryData.decompressRatio && (
            <div className="score-item">
              <span className="score-rank">📤</span>
              <span className="score-series">Scan Speed (Decompression)</span>
              <span className="score-metrics">
                <span className="score-value">{summaryData.decompressRatio.toFixed(2)}x</span>
              </span>
            </div>
          )}
        </div>
        <div className="scores-explanation">
          {summaryData.explanation || `Inverse geometric mean of Vortex/Parquet ratios across ${summaryData.datasetCount || 'multiple'} datasets (higher is better)`}
        </div>
      </div>
    );
  }

  if (summaryData.type === 'compressionSize' && summaryData.meanRatio) {
    return (
      <div className="benchmark-scores-summary">
        <h3 className="scores-title">{summaryData.title || 'Compression Size Summary'}</h3>
        <div className="scores-list">
          {summaryData.minRatio && (
            <div className="score-item">
              <span className="score-rank">⬇️</span>
              <span className="score-series">Min Size Ratio</span>
              <span className="score-metrics">
                <span className="score-value">{summaryData.minRatio.toFixed(2)}x</span>
              </span>
            </div>
          )}
          <div className="score-item">
            <span className="score-rank">📊</span>
            <span className="score-series">Mean Size Ratio</span>
            <span className="score-metrics">
              <span className="score-value">{summaryData.meanRatio.toFixed(2)}x</span>
            </span>
          </div>
          {summaryData.maxRatio && (
            <div className="score-item">
              <span className="score-rank">⬆️</span>
              <span className="score-series">Max Size Ratio</span>
              <span className="score-metrics">
                <span className="score-value">{summaryData.maxRatio.toFixed(2)}x</span>
              </span>
            </div>
          )}
        </div>
        <div className="scores-explanation">
          {summaryData.explanation || `Geometric mean of Vortex/Parquet size ratios across ${summaryData.datasetCount || 'multiple'} datasets (lower is better)`}
        </div>
      </div>
    );
  }

  return null;
}
