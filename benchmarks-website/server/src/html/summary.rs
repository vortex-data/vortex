// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Group summary card rendering.
//!
//! Each [`Summary`] variant renders into a small `.benchmark-scores-summary`
//! card that lives above the chart grid. Every variant is rendered the same
//! shape — a list of `.score-item` rows — only the rank label, value, and
//! footer change.

use maud::Markup;
use maud::html;

use crate::api::Summary;

/// Render the summary card for a group, or empty markup if `summary` is
/// `None` or every variant's content list is empty.
pub(super) fn summary_markup(summary: Option<&Summary>) -> Markup {
    let Some(summary) = summary else {
        return html! {};
    };
    match summary {
        Summary::RandomAccess {
            title,
            rankings,
            explanation,
        } if !rankings.is_empty() => html! {
            section.benchmark-scores-summary aria-label=(title) {
                h3.scores-title { (title) }
                div.scores-list {
                    @for (idx, item) in rankings.iter().enumerate() {
                        div.score-item {
                            span.score-rank { "#" (idx + 1) }
                            span.score-series title=(item.name) { (item.name) }
                            span.score-metrics {
                                span.score-value { (format_time_ns(item.time)) }
                                span.score-runtime { (format!("{:.2}x", item.ratio)) }
                            }
                        }
                    }
                }
                div.scores-explanation { (explanation) }
            }
        },
        Summary::Compression {
            title,
            compress_ratio,
            decompress_ratio,
            dataset_count: _,
            explanation,
        } if compress_ratio.is_some() || decompress_ratio.is_some() => html! {
            section.benchmark-scores-summary aria-label=(title) {
                h3.scores-title { (title) }
                div.scores-list {
                    @if let Some(v) = compress_ratio {
                        div.score-item {
                            span.score-rank { "⚡" }
                            span.score-series { "Write Speed (Compression)" }
                            span.score-metrics {
                                span.score-value { (format!("{v:.2}x")) }
                            }
                        }
                    }
                    @if let Some(v) = decompress_ratio {
                        div.score-item {
                            span.score-rank { "📤" }
                            span.score-series { "Scan Speed (Decompression)" }
                            span.score-metrics {
                                span.score-value { (format!("{v:.2}x")) }
                            }
                        }
                    }
                }
                div.scores-explanation { (explanation) }
            }
        },
        Summary::CompressionSize {
            title,
            min_ratio,
            mean_ratio,
            max_ratio,
            dataset_count: _,
            explanation,
        } => html! {
            section.benchmark-scores-summary aria-label=(title) {
                h3.scores-title { (title) }
                div.scores-list {
                    div.score-item {
                        span.score-rank { "⬇️" }
                        span.score-series { "Min Size Ratio" }
                        span.score-metrics {
                            span.score-value { (format!("{min_ratio:.2}x")) }
                        }
                    }
                    div.score-item {
                        span.score-rank { "📊" }
                        span.score-series { "Mean Size Ratio" }
                        span.score-metrics {
                            span.score-value { (format!("{mean_ratio:.2}x")) }
                        }
                    }
                    div.score-item {
                        span.score-rank { "⬆️" }
                        span.score-series { "Max Size Ratio" }
                        span.score-metrics {
                            span.score-value { (format!("{max_ratio:.2}x")) }
                        }
                    }
                }
                div.scores-explanation { (explanation) }
            }
        },
        Summary::QueryBenchmark {
            title,
            rankings,
            explanation,
        } if !rankings.is_empty() => html! {
            section.benchmark-scores-summary aria-label=(title) {
                h3.scores-title { (title) }
                div.scores-list {
                    @for (idx, item) in rankings.iter().enumerate() {
                        div.score-item {
                            span.score-rank { "#" (idx + 1) }
                            span.score-series title=(item.name) { (item.name) }
                            span.score-metrics {
                                span.score-value { (format!("{:.2}x", item.score)) }
                                span.score-runtime { (format_time_ns(item.total_runtime)) }
                            }
                        }
                    }
                }
                div.scores-explanation { (explanation) }
            }
        },
        _ => html! {},
    }
}

fn format_time_ns(ns: f64) -> String {
    let abs = ns.abs();
    if abs >= 1_000_000_000.0 {
        format!("{:.2} s", ns / 1_000_000_000.0)
    } else if abs >= 1_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else if abs >= 1_000.0 {
        format!("{:.2} us", ns / 1_000.0)
    } else {
        format!("{ns:.0} ns")
    }
}
