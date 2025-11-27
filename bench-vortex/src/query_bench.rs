// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unified benchmark infrastructure

use std::fs::File;
use std::io::Write;
use std::io::stdout;
use std::path::PathBuf;

use crate::Target;
use crate::display::DisplayFormat;
use crate::display::print_measurements_json;
use crate::display::render_table;
use crate::measurements::MemoryMeasurement;
use crate::measurements::QueryMeasurement;

/// Common benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub targets: Vec<Target>,
    pub iterations: usize,
    pub threads: Option<usize>,
    pub verbose: bool,
    pub display_format: DisplayFormat,
    pub disable_datafusion_cache: bool,
    pub queries: Option<Vec<usize>>,
    pub output_path: Option<PathBuf>,
}

/// Print benchmark results
pub fn print_results(
    display_format: &DisplayFormat,
    query_measurements: Vec<QueryMeasurement>,
    targets: &[Target],
    file_path: &Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut writer: Box<dyn Write> = if let Some(file_path) = file_path {
        Box::new(File::create(file_path)?)
    } else {
        let stdout = stdout();
        Box::new(stdout.lock())
    };

    match display_format {
        DisplayFormat::Table => render_table(&mut writer, query_measurements, targets),
        DisplayFormat::GhJson => print_measurements_json(&mut writer, query_measurements),
    }
}

/// Print memory usage
pub fn print_memory_usage(
    memory_measurements: Vec<MemoryMeasurement>,
    display_format: &DisplayFormat,
    targets: &[Target],
) -> anyhow::Result<()> {
    let mut writer = Box::new(stdout()) as Box<dyn Write>;
    match display_format {
        DisplayFormat::Table => render_table(&mut writer, memory_measurements, targets),
        DisplayFormat::GhJson => print_measurements_json(&mut writer, memory_measurements),
    }
}

/// Filter queries based on include/exclude lists
pub fn filter_queries(
    all_queries: Vec<(usize, String)>,
    include_queries: Option<&Vec<usize>>,
    exclude_queries: Option<&Vec<usize>>,
) -> Vec<(usize, String)> {
    all_queries
        .into_iter()
        .filter(|(query_idx, _)| {
            // Include query if:
            // 1. No specific queries were requested OR this query is in the requested list
            // 2. AND this query is not in the excluded list
            include_queries
                .as_ref()
                .is_none_or(|included| included.contains(query_idx))
                && exclude_queries
                    .as_ref()
                    .is_none_or(|excluded| !excluded.contains(query_idx))
        })
        .collect()
}
