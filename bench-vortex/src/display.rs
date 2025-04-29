use std::iter;

use clap::ValueEnum;
use itertools::Itertools;
use tabled::builder::Builder;
use tabled::settings::themes::Colorization;
use tabled::settings::{Color, Style};
use vortex::aliases::hash_map::HashMap;

use crate::Target;
use crate::measurements::{MeasurementValue, TableValue, ToJson, ToTable};

#[derive(ValueEnum, Default, Clone, Debug)]
pub enum DisplayFormat {
    #[default]
    Table,
    GhJson,
}

pub fn render_table<T: ToTable>(
    all_measurements: Vec<T>,
    targets: &[Target],
) -> anyhow::Result<()> {
    let mut measurements: HashMap<Target, Vec<TableValue>> =
        HashMap::with_capacity(all_measurements.len().div_ceil(targets.len()));

    let engines = targets.iter().map(|t| t.engine()).unique().collect_vec();

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
                .chain(targets.iter().map(move |t| format!("{}", t.engine())))
                .collect::<Vec<String>>(),
        );
    }

    table_builder.push_record(
        iter::once("Benchmark".to_owned())
            .chain(targets.iter().map(|t| format!("{}", t.format())))
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

    println!("{table}");

    Ok(())
}

pub fn print_measurements_json<T: ToJson>(all_measurements: Vec<T>) -> anyhow::Result<()> {
    for measurement in all_measurements {
        // This has to be `println!` and go to stdout, because we capture it from there.
        println!("{}", serde_json::to_string(&measurement.to_json())?)
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
