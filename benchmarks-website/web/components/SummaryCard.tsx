// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { formatTimeNs } from '@/lib/format';
import type { Summary } from '@/lib/summary';

/**
 * The per-group summary card, the server-component port of
 * `server/src/html/summary.rs::summary_markup`.
 *
 * Every [`Summary`] variant renders the same `.benchmark-scores-summary` shape
 * (a `.scores-title`, a `.scores-list` of `.score-item` rows, and a
 * `.scores-explanation` footer); only the rank label, value, and optional
 * runtime change. The card stays visible whether or not the enclosing group is
 * expanded (the CSS only hides `.chart-grid` when the disclosure is closed), so
 * the at-a-glance rankings show without expanding the group.
 *
 * Returns `null` when there is no summary or the variant's content list is
 * empty, matching the Rust guard arms (`if !rankings.is_empty()` etc.).
 */
export function SummaryCard({ summary }: { summary?: Summary }) {
  if (summary === undefined) {
    return null;
  }
  switch (summary.type) {
    case 'randomAccess':
      if (summary.rankings.length === 0) {
        return null;
      }
      return (
        <section className="benchmark-scores-summary" aria-label={summary.title}>
          <h3 className="scores-title">{summary.title}</h3>
          <div className="scores-list">
            {summary.rankings.map((item, idx) => (
              <div className="score-item" key={item.name}>
                <span className="score-rank">#{idx + 1}</span>
                <span className="score-series" title={item.name}>
                  {item.name}
                </span>
                <span className="score-metrics">
                  <span className="score-value">{formatTimeNs(item.time)}</span>
                  <span className="score-runtime">{item.ratio.toFixed(2)}x</span>
                </span>
              </div>
            ))}
          </div>
          <div className="scores-explanation">{summary.explanation}</div>
        </section>
      );
    case 'compression':
      if (summary.compressRatio === undefined && summary.decompressRatio === undefined) {
        return null;
      }
      return (
        <section className="benchmark-scores-summary" aria-label={summary.title}>
          <h3 className="scores-title">{summary.title}</h3>
          <div className="scores-list">
            {summary.compressRatio !== undefined && (
              <div className="score-item">
                <span className="score-rank">⚡</span>
                <span className="score-series">Write Speed (Compression)</span>
                <span className="score-metrics">
                  <span className="score-value">{summary.compressRatio.toFixed(2)}x</span>
                </span>
              </div>
            )}
            {summary.decompressRatio !== undefined && (
              <div className="score-item">
                <span className="score-rank">📤</span>
                <span className="score-series">Scan Speed (Decompression)</span>
                <span className="score-metrics">
                  <span className="score-value">{summary.decompressRatio.toFixed(2)}x</span>
                </span>
              </div>
            )}
          </div>
          <div className="scores-explanation">{summary.explanation}</div>
        </section>
      );
    case 'compressionSize':
      return (
        <section className="benchmark-scores-summary" aria-label={summary.title}>
          <h3 className="scores-title">{summary.title}</h3>
          <div className="scores-list">
            <div className="score-item">
              <span className="score-rank">⬇️</span>
              <span className="score-series">Min Size Ratio</span>
              <span className="score-metrics">
                <span className="score-value">{summary.minRatio.toFixed(2)}x</span>
              </span>
            </div>
            <div className="score-item">
              <span className="score-rank">📊</span>
              <span className="score-series">Mean Size Ratio</span>
              <span className="score-metrics">
                <span className="score-value">{summary.meanRatio.toFixed(2)}x</span>
              </span>
            </div>
            <div className="score-item">
              <span className="score-rank">⬆️</span>
              <span className="score-series">Max Size Ratio</span>
              <span className="score-metrics">
                <span className="score-value">{summary.maxRatio.toFixed(2)}x</span>
              </span>
            </div>
          </div>
          <div className="scores-explanation">{summary.explanation}</div>
        </section>
      );
    case 'queryBenchmark':
      if (summary.rankings.length === 0) {
        return null;
      }
      return (
        <section className="benchmark-scores-summary" aria-label={summary.title}>
          <h3 className="scores-title">{summary.title}</h3>
          <div className="scores-list">
            {summary.rankings.map((item, idx) => (
              <div className="score-item" key={item.name}>
                <span className="score-rank">#{idx + 1}</span>
                <span className="score-series" title={item.name}>
                  {item.name}
                </span>
                <span className="score-metrics">
                  <span className="score-value">{item.score.toFixed(2)}x</span>
                  <span className="score-runtime">{formatTimeNs(item.totalRuntime)}</span>
                </span>
              </div>
            ))}
          </div>
          <div className="scores-explanation">{summary.explanation}</div>
        </section>
      );
    default: {
      // Exhaustiveness guard: adding a new `Summary` variant without a render
      // arm above becomes a compile error here instead of a silently blank card.
      const exhaustive: never = summary;
      return exhaustive;
    }
  }
}
