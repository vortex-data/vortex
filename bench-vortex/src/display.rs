use clap::ValueEnum;
use tabled::builder::Builder;
use tabled::settings::themes::Colorization;
use tabled::settings::{Color, Style};
use vortex::aliases::hash_map::HashMap;

use crate::Format;
use crate::measurements::{MeasurementValue, TableValue, ToJson, ToTable};

#[derive(ValueEnum, Default, Clone, Debug)]
pub enum DisplayFormat {
    #[default]
    Table,
    GhJson,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RatioMode {
    Time,
    Throughput,
}

pub fn render_table<T: ToTable>(
    all_measurements: Vec<T>,
    formats: &[Format],
    mode: RatioMode,
    engine: &Option<String>,
) -> anyhow::Result<()> {
    let mut measurements: HashMap<Format, Vec<TableValue>> =
        HashMap::with_capacity(all_measurements.len().div_ceil(formats.len()));

    for m in all_measurements.into_iter() {
        let generic = m.to_table();
        measurements
            .entry(generic.format)
            .or_default()
            .push(generic);
    }

    measurements.values_mut().for_each(|v| {
        v.sort_unstable();
    });

    // The first format serves as the baseline
    let baseline_format = &formats[0];
    let baseline = measurements[baseline_format].clone();

    let mut table_builder = Builder::default();
    let mut colors = vec![];

    let mut header = vec!["Benchmark".to_owned()];
    header.extend(
        formats
            .iter()
            .map(|f| format!("{} {}", f.name(), engine.clone().unwrap_or_default())),
    );
    table_builder.push_record(header);

    for (idx, baseline_measure) in baseline.iter().enumerate() {
        let query_baseline = baseline_measure.value;
        let mut row = vec![baseline_measure.name.clone()];
        for (col_idx, format) in formats.iter().enumerate() {
            let measurement = &measurements[format][idx];
            let value = measurement.value;

            if format != baseline_format {
                let color = color(query_baseline, value, mode);

                colors.push(Colorization::exact(vec![color], (idx + 1, col_idx + 1)))
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

fn color(baseline: MeasurementValue, value: MeasurementValue, mode: RatioMode) -> Color {
    match mode {
        RatioMode::Time => {
            if value > (baseline + baseline / 2) {
                Color::BG_RED
            } else if value > (baseline + baseline / 10) {
                Color::BG_YELLOW
            } else {
                Color::BG_BRIGHT_GREEN
            }
        }
        RatioMode::Throughput => {
            if value < (baseline - baseline / 2) {
                Color::BG_RED
            } else if value < (baseline - baseline / 10) {
                Color::BG_YELLOW
            } else {
                Color::BG_BRIGHT_GREEN
            }
        }
    }
}
