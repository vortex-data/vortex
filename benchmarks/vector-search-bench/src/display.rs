// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Local table renderer for the vector-search benchmark.
//!
//! Groups columns by **flavor** (`vortex-uncompressed`, `vortex-turboquant`) rather than by
//! [`vortex_bench::Format`], because the two Vortex flavors share a single
//! `Format::OnDiskVortex`/`Format::VortexLossy` pair and the generic
//! [`vortex_bench::display::render_table`] groups by Format. Local renderer keeps the
//! column-per-flavor invariant intact without introducing a new global Format value.
//!
//! Output rows:
//!
//! ```text
//!   Metric             | vortex-uncompressed | vortex-turboquant
//!   ------------------ + ------------------- + -----------------
//!   scan wall (mean)   |              485 ms |            212 ms
//!   scan wall (median) |              490 ms |            215 ms
//!   matches            |                  42 |                39
//!   rows scanned       |          10,000,000 |        10,000,000
//!   bytes scanned      |             30.5 GB |           7.62 GB
//!   rows / sec         |              5.2e6  |           1.2e7
//! ```

use std::io::Write;

use anyhow::Result;
use tabled::settings::Style;

use crate::compression::VectorFlavor;
use crate::prepare::CompressedVortexDataset;
use crate::scan::ScanTiming;

/// Final column-per-flavor row set for one dataset.
pub struct DatasetReport<'a> {
    pub dataset_name: &'a str,
    pub vortex_results: &'a [(VectorFlavor, &'a CompressedVortexDataset, &'a ScanTiming)],
}

/// Render the full report into the given writer as a tabled table.
pub fn render(report: &DatasetReport<'_>, writer: &mut dyn Write) -> Result<()> {
    let mut headers: Vec<String> = vec!["metric".to_owned()];
    for &(flavor, ..) in report.vortex_results {
        headers.push(flavor.label().to_owned());
    }

    let rows: Vec<Vec<String>> = vec![
        make_row("scan wall (mean)", report, |_, _, scan| {
            format_duration(scan.mean)
        }),
        make_row("scan wall (median)", report, |_, _, scan| {
            format_duration(scan.median)
        }),
        make_row("matches", report, |_, _, scan| scan.matches.to_string()),
        make_row("rows scanned", report, |_, _, scan| {
            scan.rows_scanned.to_string()
        }),
        make_row("bytes scanned", report, |_, _, scan| {
            format_bytes(scan.bytes_scanned)
        }),
        make_row("rows / sec", report, |_, _, scan| {
            format_throughput_rows(scan.rows_scanned, scan.mean)
        }),
    ];

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

fn make_row<F>(metric: &str, report: &DatasetReport<'_>, vortex_cell: F) -> Vec<String>
where
    F: Fn(VectorFlavor, &CompressedVortexDataset, &ScanTiming) -> String,
{
    let mut row = vec![metric.to_owned()];
    for &(flavor, prep, scan) in report.vortex_results {
        row.push(vortex_cell(flavor, prep, scan));
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

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = UNITS[0];
    for next in &UNITS[1..] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = next;
    }
    if unit == "B" {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {unit}")
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
