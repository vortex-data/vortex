// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroU64;

use serde::Serialize;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::website::commit_info::CommitInfo;
use crate::website::entry::BenchmarkEntry;
use crate::website::entry::CommitValueMap;

/// The complete response containing benchmarks and commit metadata.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResponse {
    /// Benchmarks grouped by group name, chart name, and series name.
    pub benchmarks: Benchmarks,
    /// Sorted list of commits (by timestamp).
    pub commits: Vec<CommitInfo>,
}

/// A map of group names to their benchmark data.
pub type Benchmarks = HashMap<String, BenchmarkGroupData>;

/// Benchmark group data.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkGroupData {
    /// The name of a chart and its associated data.
    pub charts: HashMap<String, ChartData>,
}

/// Chart data.
#[derive(Debug, Clone, Serialize)]
pub struct ChartData {
    /// The name of a series and its associated data.
    pub aligned_series: HashMap<String, Vec<Option<NonZeroU64>>>,
}

// ============================================================================
// Summary data structures (for fast initial load)
// ============================================================================

/// Summary of all benchmarks (metadata only, no series values).
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkSummary {
    /// Sorted list of commits.
    pub commits: Vec<CommitInfo>,
    /// Groups with their chart and series metadata.
    pub groups: HashMap<String, GroupSummary>,
}

/// Summary of a benchmark group (metadata only).
#[derive(Debug, Clone, Serialize)]
pub struct GroupSummary {
    /// Charts in this group with their series names.
    pub charts: HashMap<String, ChartSummary>,
}

/// Summary of a chart (metadata only).
#[derive(Debug, Clone, Serialize)]
pub struct ChartSummary {
    /// Names of series in this chart.
    pub series_names: Vec<String>,
}

/// Processes benchmark entries into a structured format aligned with commits.
///
/// Series, charts, and groups with no data are automatically pruned from the result.
///
/// # Errors
///
/// Returns an error if:
/// - Commits are not sorted by timestamp.
/// - Any group, chart, or series name is empty.
/// - Series lengths don't match the number of commits (internal error).
pub fn process_benchmarks(
    entries: &[BenchmarkEntry],
    sorted_commits: &[CommitInfo],
) -> VortexResult<Benchmarks> {
    if !sorted_commits.is_sorted() {
        vortex_bail!("Commits must be sorted by timestamp");
    }

    let num_commits = sorted_commits.len();
    let grouped_entries = BenchmarkEntry::group(entries);

    let mut benchmarks = HashMap::with_capacity(grouped_entries.keys().len());
    for (group_name, group_data) in grouped_entries {
        if group_name.is_empty() {
            vortex_bail!("Group name cannot be empty");
        }

        let mut charts = HashMap::with_capacity(group_data.keys().len());
        for (chart_name, chart_data) in group_data {
            if chart_name.is_empty() {
                vortex_bail!("Chart name cannot be empty in group '{group_name}'");
            }

            let mut aligned_series = HashMap::with_capacity(chart_data.keys().len());
            for (series_name, series_data) in chart_data {
                if series_name.is_empty() {
                    vortex_bail!(
                        "Series name cannot be empty in group '{group_name}', chart '{chart_name}'",
                    );
                }

                let aligned_series_data = create_aligned_series_data(series_data, sorted_commits);

                if aligned_series_data.len() != num_commits {
                    vortex_bail!(
                        "Series '{series_name}' in group '{group_name}', chart '{chart_name}' has \
                            {} elements, expected {num_commits}",
                        aligned_series_data.len(),
                    );
                }

                // Skip series with no data points.
                if !aligned_series_data.iter().any(|v| v.is_some()) {
                    continue;
                }

                // Convert to owned String key.
                aligned_series.insert(series_name.to_string(), aligned_series_data);
            }

            if aligned_series.is_empty() {
                continue;
            }

            charts.insert(chart_name.to_string(), ChartData { aligned_series });
        }

        if charts.is_empty() {
            continue;
        }

        benchmarks.insert(group_name.to_string(), BenchmarkGroupData { charts });
    }

    Ok(benchmarks)
}

fn create_aligned_series_data(
    commits_and_values: CommitValueMap<'_>,
    sorted_commits: &[CommitInfo],
) -> Vec<Option<NonZeroU64>> {
    sorted_commits
        .iter()
        .map(|commit_info| {
            commits_and_values
                .get(commit_info.commit_id())
                .map(|&value| {
                    NonZeroU64::new(value).unwrap_or_else(|| {
                        eprintln!("Warning: benchmark value of 0 encountered, converting to 1");
                        NonZeroU64::MIN
                    })
                })
        })
        .collect()
}

/// Extracts summary metadata from benchmarks.
pub fn extract_summary(benchmarks: &Benchmarks, commits: Vec<CommitInfo>) -> BenchmarkSummary {
    let groups = benchmarks
        .iter()
        .map(|(group_name, group_data)| {
            let charts = group_data
                .charts
                .iter()
                .map(|(chart_name, chart_data)| {
                    let series_names: Vec<String> =
                        chart_data.aligned_series.keys().cloned().collect();
                    (chart_name.clone(), ChartSummary { series_names })
                })
                .collect();
            (group_name.clone(), GroupSummary { charts })
        })
        .collect();

    BenchmarkSummary { commits, groups }
}
