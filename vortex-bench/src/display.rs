// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::iter;

use anyhow::Result;
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
    baseline: Option<&HashMap<(u32, String), u64>>,
) -> Result<()> {
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

    let first_target = &targets[0];
    let reference = measurements[first_target].clone();

    let mut table_builder = Builder::default();
    let mut colors = vec![];

    let header_count = if engines.len() > 1 { 2 } else { 1 };

    if engines.len() > 1 {
        table_builder.push_record(
            iter::once("".to_owned())
                .chain(targets.iter().flat_map(|t| {
                    let label = format!("{}", t.engine);
                    if baseline.is_some() {
                        vec![label, String::new()]
                    } else {
                        vec![label]
                    }
                }))
                .collect::<Vec<String>>(),
        );
    }

    table_builder.push_record(
        iter::once("Benchmark".to_owned())
            .chain(targets.iter().flat_map(|t| {
                if baseline.is_some() {
                    vec![
                        format!("{} (baseline)", t.format),
                        format!("{} (current)", t.format),
                    ]
                } else {
                    vec![format!("{}", t.format)]
                }
            }))
            .collect::<Vec<String>>(),
    );

    let mut row = Vec::with_capacity(1 + targets.len() * (1 + baseline.is_some() as usize));
    for (row_idx, ref_m) in reference.iter().enumerate() {
        row.clear();
        row.push(ref_m.name.clone());

        if let Some(baseline) = baseline {
            let query_id = ref_m.id.map(|i| i as u32);
            for (target_col, target) in targets.iter().enumerate() {
                let measurement = &measurements[target][row_idx];
                let value = measurement.value;
                // baseline stores nanoseconds, TableValue uses microseconds.
                let bv_us = query_id.and_then(|id| {
                    baseline
                        .get(&(id, target.format.name().to_string()))
                        .map(|&ns| ns / 1_000)
                });
                let Some(bv_us) = bv_us else {
                    // We have already filtered missing values in
                    // build_query_baseline_map
                    anyhow::bail!("Query id or baseline value missing");
                };

                assert!(bv_us > 0);
                let bv = MeasurementValue::Int(bv_us as u128);
                let ratio = value / bv;
                row.push(format!("{bv:.2} {}", measurement.unit));
                row.push(format!("{value:.2} {} ({ratio:.2})", measurement.unit));
                colors.push(Colorization::exact(
                    vec![color(bv, value)],
                    (row_idx + header_count, 2 + target_col * 2),
                ));
            }
        } else {
            let query_baseline = ref_m.value;
            for (col_idx, target) in targets.iter().enumerate() {
                let measurement = &measurements[target][row_idx];
                let value = measurement.value;
                if target != first_target {
                    colors.push(Colorization::exact(
                        vec![color(query_baseline, value)],
                        (row_idx + header_count, col_idx + 1),
                    ));
                }
                let ratio = value / query_baseline;
                row.push(format!("{value:.2} {} ({ratio:.2})", measurement.unit));
            }
        }

        table_builder.push_record(&row);
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
) -> Result<()> {
    for measurement in all_measurements {
        writeln!(writer, "{}", measurement.to_json())?;
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
