// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::iter;

use clap::ValueEnum;
use itertools::Itertools;
use tabled::builder::Builder;
use tabled::settings::Color;
use tabled::settings::Style;
use tabled::settings::themes::Colorization;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Target;
use crate::measurements::MeasurementValue;
use crate::measurements::TableValue;
use crate::measurements::ToJson;
use crate::measurements::ToTable;

#[derive(ValueEnum, Default, Clone, Debug)]
pub enum DisplayFormat {
    #[default]
    Table,
    GhJson,
}

pub fn render_table<W: Write, T: ToTable>(
    writer: &mut W,
    all_measurements: Vec<T>,
    targets: &[Target],
) -> anyhow::Result<()> {
    // `all_measurements` is empty for decompress-only runs such as `compress-bench
    // --gpu-decompress`: every ratio compares vortex against parquet or lance, neither of
    // which is benchmarked there, so no ratio is recorded even though a baseline target is
    // still passed. `targets` is instead empty when the caller has no baseline format to
    // report against. Either way there is no table to draw, and the indexing below would
    // panic looking up a baseline that has no measurements.
    if all_measurements.is_empty() || targets.is_empty() {
        return Ok(());
    }

    let mut measurements: HashMap<Target, Vec<TableValue>> =
        HashMap::with_capacity(all_measurements.len().div_ceil(targets.len()));

    let engines = targets.iter().map(|t| t.engine).unique().collect_vec();

    for m in all_measurements.into_iter() {
        let generic = m.to_table();
        measurements
            .entry(generic.target)
            .or_default()
            .push(generic);
    }

    measurements.values_mut().sorted_unstable();

    // The first format serves as the baseline
    let baseline_target = &targets[0];
    let baseline = measurements[baseline_target].clone();

    let mut table_builder = Builder::default();
    let mut colors = vec![];

    let header_count = if engines.len() > 1 { 2 } else { 1 };

    if engines.len() > 1 {
        table_builder.push_record(
            iter::once("".to_owned())
                .chain(targets.iter().map(move |t| format!("{}", t.engine)))
                .collect::<Vec<String>>(),
        );
    }

    table_builder.push_record(
        iter::once("Benchmark".to_owned())
            .chain(targets.iter().map(|t| format!("{}", t.format)))
            .collect::<Vec<String>>(),
    );

    for (idx, baseline_measure) in baseline.iter().enumerate() {
        let query_baseline = baseline_measure.value;
        let mut row = vec![baseline_measure.name.clone()];
        for (col_idx, target) in targets.iter().enumerate() {
            let measurement = &measurements[target][idx];
            let value = measurement.value;

            if target != baseline_target {
                let color = color(query_baseline, value);

                colors.push(Colorization::exact(
                    vec![color],
                    (idx + header_count, col_idx + 1),
                ))
            }

            let ratio = value / query_baseline;
            row.push(format!("{value:.2} {} ({ratio:.2})", measurement.unit));
        }
        table_builder.push_record(row);
    }

    let mut table = table_builder.build();
    table.with(Style::modern());

    for color in colors.into_iter() {
        table.with(color);
    }

    writeln!(writer, "{table}")?;

    Ok(())
}

pub fn print_measurements_json<T: ToJson>(
    writer: &mut dyn Write,
    all_measurements: Vec<T>,
    doc: &str,
) -> anyhow::Result<()> {
    for measurement in all_measurements {
        let mut json = measurement.to_json();
        if let Some(obj) = json.as_object_mut() {
            obj.insert("doc".to_string(), doc.into());
        }
        writeln!(writer, "{json}")?;
    }

    Ok(())
}

fn color(baseline: MeasurementValue, value: MeasurementValue) -> Color {
    if value > (baseline + baseline / 2) {
        Color::BG_RED | Color::FG_BLACK
    } else if value > (baseline + baseline / 10) {
        Color::BG_YELLOW | Color::FG_BLACK
    } else {
        Color::BG_BRIGHT_GREEN | Color::FG_BLACK
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::Engine;
    use crate::Format;
    use crate::measurements::CompressionTimingMeasurement;

    /// Decompress-only runs (`compress-bench --gpu-decompress`) collect no compression ratios,
    /// and callers pass an empty target list when the baseline format is absent. Both used to
    /// panic: the first on the missing baseline key, the second on a divide-by-zero capacity.
    #[rstest]
    #[case::no_measurements(&[Target::new(Engine::Vortex, Format::OnDiskVortex)])]
    #[case::no_targets(&[])]
    fn render_table_without_a_baseline_is_a_noop(#[case] targets: &[Target]) -> anyhow::Result<()> {
        let mut out = Vec::new();
        render_table(
            &mut out,
            Vec::<CompressionTimingMeasurement>::new(),
            targets,
        )?;
        assert!(out.is_empty());
        Ok(())
    }
}
