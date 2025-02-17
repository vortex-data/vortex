use clap::ValueEnum;
use tabled::builder::Builder;
use tabled::settings::themes::Colorization;
use tabled::settings::{Color, Style};
use vortex::aliases::hash_map::HashMap;

use crate::measurements::{GenericMeasurement, ToGeneric, ToJson};
use crate::Format;

#[derive(ValueEnum, Default, Clone, Debug)]
pub enum DisplayFormat {
    #[default]
    Table,
    GhJson,
}

pub fn render_table<T: ToGeneric>(
    all_measurements: Vec<T>,
    formats: &[Format],
) -> anyhow::Result<()> {
    let mut measurements: HashMap<Format, Vec<GenericMeasurement>> = HashMap::default();

    for m in all_measurements.into_iter() {
        let generic = m.to_generic();
        measurements
            .entry(generic.format)
            .or_default()
            .push(generic);
    }

    measurements.values_mut().for_each(|v| {
        v.sort_by_key(|m| m.id);
    });

    // The first format serves as the baseline
    let baseline_format = &formats[0];
    let baseline = measurements[baseline_format].clone();

    let mut table_builder = Builder::default();
    let mut colors = vec![];

    let mut header = vec!["Query".to_string()];
    header.extend(formats.iter().map(|f| format!("{:?}", f)));
    table_builder.push_record(header);

    for (idx, baseline_measure) in baseline.iter().enumerate() {
        let query_baseline = baseline_measure.time.as_micros();
        let mut row = vec![baseline_measure.id.to_string()];
        for (col_idx, format) in formats.iter().enumerate() {
            let time_us = measurements[format][idx].time.as_micros();

            if format != baseline_format {
                let color = color(query_baseline, time_us);

                colors.push(Colorization::exact(vec![color], (idx + 1, col_idx + 1)))
            }

            let ratio = time_us as f64 / query_baseline as f64;
            row.push(format!("{time_us} us ({ratio:.2})"));
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
        println!("{}", serde_json::to_string(&measurement.to_json())?)
    }

    Ok(())
}

fn color(baseline_time: u128, test_time: u128) -> Color {
    if test_time > (baseline_time + baseline_time / 2) {
        Color::BG_RED
    } else if test_time > (baseline_time + baseline_time / 10) {
        Color::BG_YELLOW
    } else {
        Color::BG_BRIGHT_GREEN
    }
}
