// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Variant-major table renderer for `vector-search-bench`.
//!
//! Unlike `compress-bench` and `random-access-bench` — which can lean on the
//! shared [`vortex_bench::display::render_table`] because their measurements
//! are keyed by `(engine, format)` and every column has the same set of row
//! names — `vector-search-bench` runs multiple "variants" that collapse to
//! the same [`vortex_bench::Format`] enum value (e.g. `vortex-uncompressed`
//! and `vortex-default` both report as `Format::OnDiskVortex`). The generic
//! helper would merge them into one column, so this module renders a table
//! keyed by **variant label** instead of `Target`.
//!
//! The output shape deliberately mirrors
//! [`vortex_bench::display::render_table`]: one column per variant (in the
//! order the user passed on `--formats`, with the first variant that
//! actually has a value serving as the ratio baseline for each row),
//! `tabled::Style::modern`, and per-cell green/yellow/red coloring based on
//! the ratio against the baseline.
//!
//! This module is only used for the `-d table` developer-inspection path.
//! The `-d gh-json` path (the one CI consumes) lives entirely in
//! [`vortex_bench::display::print_measurements_json`] and the `ToJson`
//! impls on the measurement structs, both of which are untouched.

use std::io::Write;

use anyhow::Result;
use tabled::builder::Builder;
use tabled::settings::Color;
use tabled::settings::Style;
use tabled::settings::themes::Colorization;
use vortex_bench::measurements::CompressionTimingMeasurement;
use vortex_bench::measurements::CustomUnitMeasurement;

/// How a row's raw `f64` values are rendered in a cell.
///
/// This axis is "the physical quantity being reported", not "the metric
/// name" — multiple metric names may share a [`RowFormat`]. The ratio
/// calculation always runs over the raw `f64`, so ratios stay comparable
/// regardless of how the value is displayed.
#[derive(Clone, Copy, Debug)]
pub enum RowFormat {
    /// Raw value is nanoseconds; render as `"{x:.2} ms"`.
    DurationNanos,
    /// Raw value is bytes; render as `"{x:.2} MB"`.
    Bytes,
    /// Dimensionless absolute difference; render in scientific notation.
    AbsDiff,
    /// Recall fraction in `[0, 1]`; render with four decimal places.
    Recall,
}

impl RowFormat {
    fn format(self, value: f64) -> String {
        match self {
            Self::DurationNanos => format!("{:.2} ms", value / 1_000_000.0),
            Self::Bytes => format!("{:.2} MB", value / (1024.0 * 1024.0)),
            Self::AbsDiff => format!("{value:.2e}"),
            Self::Recall => format!("{value:.4}"),
        }
    }
}

/// One row of the rendered table: a label plus one cell per variant column.
///
/// `cells` is parallel to the `variant_labels` slice passed to
/// [`render_variants_table`]; `None` means "no measurement for this variant"
/// (e.g. the `handrolled` variant has no `compress time`) and renders as an
/// empty `-` cell.
#[derive(Clone, Debug)]
pub struct Row {
    /// Row label, typically `"{metric_kind}/{dataset_name}"`.
    pub label: String,
    /// How to render the raw `f64` values in this row's cells.
    pub format: RowFormat,
    /// One entry per variant column, in the same order as `variant_labels`.
    pub cells: Vec<Option<f64>>,
}

/// Render a variant-major table of `rows` to `writer`.
///
/// `variant_labels` fixes the column order and must match the cell order in
/// every [`Row`]. `preferred_baseline` names the variant whose column should
/// be used as the ratio baseline: when the named column has a value for a
/// given row, that value is the baseline; otherwise the function falls back
/// to the first column whose cell is `Some`. `preferred_baseline = None`
/// (or a label that isn't in `variant_labels`) produces pure first-non-None
/// behavior.
///
/// Baseline per row rather than per table matters because some rows have a
/// legitimate missing-cell pattern: e.g. `compress time` has no value for
/// `handrolled`, so even with `preferred_baseline = Some("handrolled")`
/// that row has to fall back to a non-handrolled column.
///
/// The styling (`Style::modern`, three-tier ratio coloring) matches
/// [`vortex_bench::display::render_table`] so a human looking at both
/// tables side-by-side sees the same formatting conventions.
pub fn render_variants_table<W: Write>(
    writer: &mut W,
    variant_labels: &[String],
    rows: &[Row],
    preferred_baseline: Option<&str>,
) -> Result<()> {
    // Resolve the preferred baseline label to a column index once up front;
    // if the user passed a variant that isn't in this run's column set we
    // just degrade to "first non-None" for every row.
    let preferred_col =
        preferred_baseline.and_then(|label| variant_labels.iter().position(|v| v == label));

    let mut builder = Builder::default();
    builder.push_record(
        std::iter::once("Benchmark".to_owned())
            .chain(variant_labels.iter().cloned())
            .collect::<Vec<_>>(),
    );

    let mut colors = Vec::new();
    for (row_idx, row) in rows.iter().enumerate() {
        // Header is row 0, so data rows start at row_idx + 1.
        let table_row_idx = row_idx + 1;

        // Prefer the caller-supplied baseline column for this row if it has
        // a value; otherwise fall back to the first column that does.
        let baseline_col = preferred_col
            .filter(|&c| row.cells.get(c).copied().flatten().is_some())
            .or_else(|| row.cells.iter().position(Option::is_some));
        let baseline_value = baseline_col.and_then(|i| row.cells[i]);

        let mut record = vec![row.label.clone()];
        for (col_idx, cell) in row.cells.iter().enumerate() {
            match (cell, baseline_value) {
                (None, _) => record.push("-".to_owned()),
                (Some(value), Some(baseline))
                    if Some(col_idx) != baseline_col && baseline > 0.0 =>
                {
                    let ratio = value / baseline;
                    // Data columns are offset by 1 because of the label column.
                    colors.push(Colorization::exact(
                        vec![ratio_color(ratio)],
                        (table_row_idx, col_idx + 1),
                    ));
                    record.push(format!("{} ({:.2})", row.format.format(*value), ratio));
                }
                (Some(value), _) => record.push(row.format.format(*value)),
            }
        }
        builder.push_record(record);
    }

    let mut table = builder.build();
    table.with(Style::modern());
    for color in colors {
        table.with(color);
    }

    writeln!(writer, "{table}")?;
    Ok(())
}

/// Three-tier coloring matching `vortex_bench::display::color`.
///
/// Green for "within 10% of baseline" (likely equal or better accounting
/// for noise), yellow for "up to 50% slower", red for "more than 50%
/// slower". Only applied to non-baseline cells with a computed ratio.
fn ratio_color(ratio: f64) -> Color {
    if ratio > 1.5 {
        Color::BG_RED | Color::FG_BLACK
    } else if ratio > 1.1 {
        Color::BG_YELLOW | Color::FG_BLACK
    } else {
        Color::BG_BRIGHT_GREEN | Color::FG_BLACK
    }
}

/// Assemble the [`Row`] list fed into [`render_variants_table`] by walking
/// the already-collected measurement vecs.
///
/// This function is deliberately **additive over** the measurement-push
/// code in `main.rs` — it doesn't touch how measurements are collected,
/// serialized, or shipped to CI. Row-cell lookup reconstructs the exact
/// metric-name strings the push-side code uses so there's no fragile
/// substring parsing: we already know the grammar because we pass the
/// same `variant_label` strings both to the push side and to this
/// builder. A row whose cells are all `None` for every variant is
/// suppressed via `push_row_if_any` (e.g. the `recall@k` row when
/// `--recall-queries 0`, or the `compress time` row when no vortex
/// variant is active).
pub fn build_table_rows(
    variant_labels: &[String],
    dataset_names: &[String],
    recall_k: usize,
    timings: &[CompressionTimingMeasurement],
    sizes: &[CustomUnitMeasurement],
    recalls: &[CustomUnitMeasurement],
    verification: &[CustomUnitMeasurement],
) -> Vec<Row> {
    let mut rows = Vec::new();

    for dataset in dataset_names {
        // Size — `handrolled` labels its size as `"{variant} size/{dataset}"`
        // (parquet bytes on disk) while vortex variants use
        // `"{variant} nbytes/{dataset}"` (in-memory nbytes). They're
        // different quantities but both represent "how big is this
        // variant", so they share one row.
        let size_cells: Vec<Option<f64>> = variant_labels
            .iter()
            .map(|variant| {
                let disk = format!("{variant} size/{dataset}");
                let nbytes = format!("{variant} nbytes/{dataset}");
                sizes
                    .iter()
                    .find(|m| m.name == disk || m.name == nbytes)
                    .map(|m| m.value)
            })
            .collect();
        push_row_if_any(
            &mut rows,
            format!("size/{dataset}"),
            RowFormat::Bytes,
            size_cells,
        );

        // Timing rows — same `{metric} time/{variant}/{dataset}` grammar
        // for every timing metric. `compress time` is legitimately absent
        // for `handrolled`; `push_row_if_any` handles the fully-missing
        // case (e.g. `--formats handrolled` alone) by skipping the row.
        for metric in [
            "compress",
            "decompress",
            "cosine-similarity",
            "cosine-filter",
        ] {
            let cells: Vec<Option<f64>> = variant_labels
                .iter()
                .map(|variant| {
                    let name = format!("{metric} time/{variant}/{dataset}");
                    timings
                        .iter()
                        .find(|t| t.name == name)
                        .map(|t| t.time.as_nanos() as f64)
                })
                .collect();
            push_row_if_any(
                &mut rows,
                format!("{metric} time/{dataset}"),
                RowFormat::DurationNanos,
                cells,
            );
        }

        // Correctness — every variant emits this.
        let correctness_cells: Vec<Option<f64>> = variant_labels
            .iter()
            .map(|variant| {
                let name = format!("correctness-max-diff/{variant}/{dataset}");
                verification
                    .iter()
                    .find(|m| m.name == name)
                    .map(|m| m.value)
            })
            .collect();
        push_row_if_any(
            &mut rows,
            format!("correctness-max-diff/{dataset}"),
            RowFormat::AbsDiff,
            correctness_cells,
        );

        // Recall — only `vortex-turboquant` emits this, and only when
        // `--recall-queries > 0`. The row is skipped entirely when no
        // variant has a value (which is the common dev case of passing
        // `--recall-queries 0` for fast iteration).
        let recall_cells: Vec<Option<f64>> = variant_labels
            .iter()
            .map(|variant| {
                let name = format!("recall@{recall_k}/{variant}/{dataset}");
                recalls.iter().find(|m| m.name == name).map(|m| m.value)
            })
            .collect();
        push_row_if_any(
            &mut rows,
            format!("recall@{recall_k}/{dataset}"),
            RowFormat::Recall,
            recall_cells,
        );
    }

    rows
}

/// Push a row into `rows` only if at least one cell has a value.
///
/// Used to suppress entirely-empty rows like `compress time` when the user
/// passes `--formats handrolled` (no vortex variants) or `recall@k` when
/// `--recall-queries 0`.
fn push_row_if_any(rows: &mut Vec<Row>, label: String, format: RowFormat, cells: Vec<Option<f64>>) {
    if cells.iter().any(Option::is_some) {
        rows.push(Row {
            label,
            format,
            cells,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_format_formats_duration_as_ms() {
        assert_eq!(RowFormat::DurationNanos.format(1_500_000.0), "1.50 ms");
        assert_eq!(RowFormat::DurationNanos.format(0.0), "0.00 ms");
    }

    #[test]
    fn row_format_formats_bytes_as_mb() {
        assert_eq!(RowFormat::Bytes.format(1_048_576.0), "1.00 MB");
        assert_eq!(RowFormat::Bytes.format(314_572_800.0), "300.00 MB");
    }

    #[test]
    fn row_format_formats_absdiff_scientific() {
        assert_eq!(RowFormat::AbsDiff.format(0.005459249), "5.46e-3");
        assert_eq!(RowFormat::AbsDiff.format(0.0), "0.00e0");
    }

    #[test]
    fn row_format_formats_recall_fourdp() {
        assert_eq!(RowFormat::Recall.format(0.987654321), "0.9877");
        assert_eq!(RowFormat::Recall.format(1.0), "1.0000");
    }

    #[test]
    fn render_smoke_two_variants_one_row() -> Result<()> {
        let variants = vec!["handrolled".to_owned(), "vortex-uncompressed".to_owned()];
        let rows = vec![Row {
            label: "decompress time/cohere-small".to_owned(),
            format: RowFormat::DurationNanos,
            cells: vec![Some(2_000_000.0), Some(4_000_000.0)],
        }];

        let mut out = Vec::new();
        render_variants_table(&mut out, &variants, &rows, None)?;
        let rendered = String::from_utf8(out).expect("table is utf8");

        assert!(rendered.contains("Benchmark"), "missing header: {rendered}");
        assert!(
            rendered.contains("handrolled"),
            "missing variant: {rendered}"
        );
        assert!(
            rendered.contains("vortex-uncompressed"),
            "missing variant: {rendered}"
        );
        assert!(
            rendered.contains("decompress time/cohere-small"),
            "missing row label: {rendered}"
        );
        assert!(
            rendered.contains("2.00 ms"),
            "missing baseline cell: {rendered}"
        );
        assert!(
            rendered.contains("4.00 ms"),
            "missing non-baseline cell: {rendered}"
        );
        assert!(
            rendered.contains("(2.00)"),
            "missing ratio annotation: {rendered}"
        );
        Ok(())
    }

    #[test]
    fn render_with_missing_first_cell_promotes_next_variant_to_baseline() -> Result<()> {
        // Simulates the `compress time` row: `handrolled` (first column) has
        // no value, so `vortex-uncompressed` should become the effective
        // baseline and render with no ratio, while `vortex-turboquant`
        // renders a ratio against it.
        let variants = vec![
            "handrolled".to_owned(),
            "vortex-uncompressed".to_owned(),
            "vortex-turboquant".to_owned(),
        ];
        let rows = vec![Row {
            label: "compress time/cohere-small".to_owned(),
            format: RowFormat::DurationNanos,
            cells: vec![None, Some(1_000_000.0), Some(3_000_000.0)],
        }];

        let mut out = Vec::new();
        render_variants_table(&mut out, &variants, &rows, None)?;
        let rendered = String::from_utf8(out).expect("table is utf8");

        // The missing cell renders as "-".
        assert!(
            rendered.contains(" - "),
            "missing dash placeholder: {rendered}"
        );
        // The baseline cell has no ratio annotation (the `(...)` suffix).
        assert!(
            rendered.contains("1.00 ms"),
            "missing baseline cell: {rendered}"
        );
        // The non-baseline cell has a ratio against the promoted baseline.
        assert!(
            rendered.contains("3.00 ms (3.00)"),
            "missing non-baseline ratio: {rendered}"
        );
        Ok(())
    }

    #[test]
    fn preferred_baseline_is_used_when_present() -> Result<()> {
        // With `preferred_baseline = Some("vortex-uncompressed")`, the
        // middle column becomes the ratio baseline even though there's a
        // valid value in the first column. `handrolled` (2 ms) renders as
        // "0.50" relative to `vortex-uncompressed` (4 ms), and
        // `vortex-turboquant` (8 ms) renders as "2.00".
        let variants = vec![
            "handrolled".to_owned(),
            "vortex-uncompressed".to_owned(),
            "vortex-turboquant".to_owned(),
        ];
        let rows = vec![Row {
            label: "decompress time/cohere-small".to_owned(),
            format: RowFormat::DurationNanos,
            cells: vec![Some(2_000_000.0), Some(4_000_000.0), Some(8_000_000.0)],
        }];

        let mut out = Vec::new();
        render_variants_table(&mut out, &variants, &rows, Some("vortex-uncompressed"))?;
        let rendered = String::from_utf8(out).expect("table is utf8");

        // The baseline column has no ratio suffix.
        assert!(
            rendered.contains("4.00 ms") && !rendered.contains("4.00 ms ("),
            "vortex-uncompressed should render without a ratio: {rendered}"
        );
        // Handrolled is now expressed relative to the preferred baseline.
        assert!(
            rendered.contains("2.00 ms (0.50)"),
            "handrolled should render with ratio 0.50: {rendered}"
        );
        assert!(
            rendered.contains("8.00 ms (2.00)"),
            "vortex-turboquant should render with ratio 2.00: {rendered}"
        );
        Ok(())
    }

    #[test]
    fn preferred_baseline_falls_back_when_row_has_no_preferred_value() -> Result<()> {
        // `compress time` again: preferred baseline is vortex-uncompressed,
        // but when vortex-uncompressed has no value for this row (simulated
        // with `None`), the function must fall back to the first-non-None
        // column rather than panicking or skipping the row.
        let variants = vec![
            "vortex-uncompressed".to_owned(),
            "vortex-default".to_owned(),
            "vortex-turboquant".to_owned(),
        ];
        let rows = vec![Row {
            label: "compress time/cohere-small".to_owned(),
            format: RowFormat::DurationNanos,
            cells: vec![None, Some(2_000_000.0), Some(6_000_000.0)],
        }];

        let mut out = Vec::new();
        render_variants_table(&mut out, &variants, &rows, Some("vortex-uncompressed"))?;
        let rendered = String::from_utf8(out).expect("table is utf8");

        // vortex-default is now the promoted baseline (no ratio).
        assert!(
            rendered.contains("2.00 ms") && !rendered.contains("2.00 ms ("),
            "vortex-default should render as promoted baseline: {rendered}"
        );
        // vortex-turboquant is 3x vortex-default.
        assert!(
            rendered.contains("6.00 ms (3.00)"),
            "vortex-turboquant should render ratio 3.00 vs promoted baseline: {rendered}"
        );
        Ok(())
    }

    #[test]
    fn preferred_baseline_label_not_in_variants_degrades_gracefully() -> Result<()> {
        // If the caller passes a label that isn't in the variant list
        // (e.g. `--formats handrolled,vortex-turboquant` with preferred
        // `vortex-uncompressed`), the function must not panic and should
        // fall back to first-non-None.
        let variants = vec!["handrolled".to_owned(), "vortex-turboquant".to_owned()];
        let rows = vec![Row {
            label: "decompress time/cohere-small".to_owned(),
            format: RowFormat::DurationNanos,
            cells: vec![Some(1_000_000.0), Some(2_000_000.0)],
        }];

        let mut out = Vec::new();
        render_variants_table(&mut out, &variants, &rows, Some("vortex-uncompressed"))?;
        let rendered = String::from_utf8(out).expect("table is utf8");

        // First column (handrolled) is the fallback baseline, second has a 2.00 ratio.
        assert!(
            rendered.contains("2.00 ms (2.00)"),
            "fallback baseline ratio missing: {rendered}"
        );
        Ok(())
    }
}
