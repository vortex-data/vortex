// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Local table renderer for the vector-search benchmark.
//!
//! Groups columns by **flavor** (`vortex-uncompressed`, `vortex-turboquant`, `handrolled`)
//! rather than by [`vortex_bench::Format`], because the two Vortex flavors share a single
//! `Format::OnDiskVortex`/`Format::VortexLossy` pair and the generic
//! [`vortex_bench::display::render_table`] groups by Format. Local renderer keeps the
//! column-per-flavor invariant intact without introducing a new global Format value.
//!
//! Output rows:
//!
//! ```text
//!   Metric           | vortex-uncompressed | vortex-turboquant | handrolled
//!   ---------------- + ------------------- + ----------------- + ----------
//!   compress wall    |               1.2 s |             3.4 s |        n/a
//!   output bytes     |             5.0 GiB |           1.2 GiB |   12.0 GiB (parquet)
//!   scan-wall (best) |              480 ms |            210 ms |     1.5 s
//!   scan-wall (med)  |              490 ms |            215 ms |     1.6 s
//!   matches          |                  42 |                39 |         42
//!   rows / sec       |              5.2e6  |           1.2e7   |    3.3e5
//! ```

use std::io::Write;

use anyhow::Result;
use tabled::settings::Style;

use crate::compression::VortexCompression;
use crate::handrolled::HandrolledTiming;
use crate::prepare::CompressionResult;
use crate::recall::RecallResult;
use crate::scan::ScanTiming;

/// Final column-per-flavor row set for one dataset.
pub struct DatasetReport<'a> {
    pub dataset_name: &'a str,
    pub vortex_results: &'a [(VortexCompression, &'a CompressionResult, &'a ScanTiming)],
    pub handrolled: Option<&'a HandrolledTiming>,
    /// Per-flavor recall results when `--recall` was requested. Empty otherwise.
    pub recall: &'a [RecallResult],
}

/// Render the full report into the given writer as a tabled table.
pub fn render(report: &DatasetReport<'_>, writer: &mut dyn Write) -> Result<()> {
    let mut headers: Vec<String> = vec!["metric".to_owned()];
    for &(flavor, ..) in report.vortex_results {
        headers.push(flavor.label().to_owned());
    }
    if report.handrolled.is_some() {
        headers.push("handrolled".to_owned());
    }

    let mut rows: Vec<Vec<String>> = Vec::new();

    rows.push(make_row(
        "compress wall",
        report,
        |_, prep, _| format_duration(prep.total_wall_time),
        |_| "n/a".to_owned(),
    ));
    rows.push(make_row(
        "input bytes",
        report,
        |_, prep, _| humanize_bytes(prep.total_input_bytes),
        |h| humanize_bytes(h.total_input_bytes),
    ));
    rows.push(make_row(
        "output bytes",
        report,
        |_, prep, _| humanize_bytes(prep.total_output_bytes),
        |h| format!("{} (parquet)", humanize_bytes(h.total_input_bytes)),
    ));
    rows.push(make_row(
        "compression ratio",
        report,
        |_, prep, _| {
            if prep.total_input_bytes == 0 {
                "—".to_owned()
            } else {
                format!(
                    "{:.2}x",
                    prep.total_input_bytes as f64 / prep.total_output_bytes.max(1) as f64
                )
            }
        },
        |_| "1.00x".to_owned(),
    ));
    rows.push(make_row(
        "scan wall (best)",
        report,
        |_, _, scan| format_duration(scan.best_of),
        |h| format_duration(h.best_of),
    ));
    rows.push(make_row(
        "scan wall (median)",
        report,
        |_, _, scan| format_duration(scan.median()),
        |h| format_duration(h.median()),
    ));
    rows.push(make_row(
        "matches",
        report,
        |_, _, scan| scan.matches.to_string(),
        |h| h.matches.to_string(),
    ));
    rows.push(make_row(
        "rows scanned",
        report,
        |_, _, scan| scan.rows_scanned.to_string(),
        |h| h.rows_scanned.to_string(),
    ));
    rows.push(make_row(
        "rows / sec",
        report,
        |_, _, scan| format_throughput_rows(scan.rows_scanned, scan.best_of),
        |h| format_throughput_rows(h.rows_scanned, h.best_of),
    ));

    if !report.recall.is_empty() {
        let k = report.recall[0].k;
        rows.push(make_row(
            &format!("recall@{k} (mean)"),
            report,
            |flavor, _, _| {
                report
                    .recall
                    .iter()
                    .find(|r| r.flavor == flavor)
                    .map(|r| format!("{:.3}", r.mean_recall))
                    .unwrap_or_else(|| "—".to_owned())
            },
            |_| "—".to_owned(),
        ));
        rows.push(make_row(
            &format!("recall@{k} (p05)"),
            report,
            |flavor, _, _| {
                report
                    .recall
                    .iter()
                    .find(|r| r.flavor == flavor)
                    .map(|r| format!("{:.3}", r.p05_recall))
                    .unwrap_or_else(|| "—".to_owned())
            },
            |_| "—".to_owned(),
        ));
    }

    writeln!(writer, "## {}", report.dataset_name)?;
    let mut builder = tabled::builder::Builder::new();
    builder.push_record(headers);
    for row in rows {
        builder.push_record(row);
    }
    let mut table = builder.build();
    table.with(Style::modern());
    writeln!(writer, "{table}")?;
    Ok(())
}

fn make_row<F, G>(
    metric: &str,
    report: &DatasetReport<'_>,
    vortex_cell: F,
    handrolled_cell: G,
) -> Vec<String>
where
    F: Fn(VortexCompression, &CompressionResult, &ScanTiming) -> String,
    G: Fn(&HandrolledTiming) -> String,
{
    let mut row = vec![metric.to_owned()];
    for &(flavor, prep, scan) in report.vortex_results {
        row.push(vortex_cell(flavor, prep, scan));
    }
    if let Some(h) = report.handrolled {
        row.push(handrolled_cell(h));
    }
    row
}

fn format_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.2} s")
    } else if secs >= 1e-3 {
        format!("{:.1} ms", secs * 1e3)
    } else {
        format!("{:.1} µs", secs * 1e6)
    }
}

fn humanize_bytes(bytes: u64) -> String {
    const KB: u64 = 1 << 10;
    const MB: u64 = 1 << 20;
    const GB: u64 = 1 << 30;
    if bytes >= GB {
        format!("{:.2} GiB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MiB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KiB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn format_throughput_rows(rows: u64, wall: std::time::Duration) -> String {
    let secs = wall.as_secs_f64();
    if secs <= 0.0 {
        return "—".to_owned();
    }
    let rps = rows as f64 / secs;
    if rps >= 1e9 {
        format!("{:.2}G", rps / 1e9)
    } else if rps >= 1e6 {
        format!("{:.2}M", rps / 1e6)
    } else if rps >= 1e3 {
        format!("{:.2}K", rps / 1e3)
    } else {
        format!("{rps:.0}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanize_bytes_picks_unit() {
        assert_eq!(humanize_bytes(0), "0 B");
        assert_eq!(humanize_bytes(1024), "1.00 KiB");
        assert_eq!(humanize_bytes(1024 * 1024), "1.00 MiB");
        assert_eq!(humanize_bytes(1024 * 1024 * 1024), "1.00 GiB");
    }

    #[test]
    fn format_throughput_picks_unit() {
        assert_eq!(
            format_throughput_rows(1_500_000, std::time::Duration::from_secs(1)),
            "1.50M"
        );
        assert_eq!(
            format_throughput_rows(0, std::time::Duration::from_secs(1)),
            "0"
        );
        assert_eq!(format_throughput_rows(100, std::time::Duration::ZERO), "—");
    }
}
