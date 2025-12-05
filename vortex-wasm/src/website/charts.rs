// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use serde::Deserialize;
use serde::Serialize;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::website::commit::CommitInfo;
use crate::website::entry::BenchmarkEntry;
use crate::website::entry::CommitValueMap;

/// Log to the browser console (WASM) or stderr (native).
#[cfg(target_arch = "wasm32")]
macro_rules! log {
    ($($t:tt)*) => {
        web_sys::console::log_1(&format!($($t)*).into());
    }
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! log {
    ($($t:tt)*) => {
        eprintln!($($t)*);
    }
}

/// The complete response containing benchmarks and commit metadata.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResponse<'a> {
    /// Benchmarks grouped by group name, chart name, and series name.
    pub benchmarks: Benchmarks<'a>,
    /// Sorted list of commits (by timestamp). Owned to allow serialization.
    pub commits: Vec<CommitInfo>,
}

/// A map of group names to their benchmark data.
pub type Benchmarks<'a> = HashMap<&'a str, BenchmarkGroupData<'a>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkGroupData<'a> {
    /// The name of a chart and its associated data.
    #[serde(borrow)]
    charts: HashMap<&'a str, ChartData<'a>>,
}

// TODO(connor): We should be able to use an `Option<NonZeroU64>` since our benchmarks should
// basically never hit 0, but that is an optimization for another day.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData<'a> {
    /// The name of a series and its associated data.
    #[serde(borrow)]
    aligned_series: HashMap<&'a str, Vec<Option<u64>>>,
}

// ============================================================================
// Owned data structures (for static caching)
// ============================================================================

/// A map of group names to their benchmark data (owned version for static storage).
pub type OwnedBenchmarks = HashMap<String, OwnedBenchmarkGroupData>;

/// Benchmark group data with owned strings.
#[derive(Debug, Clone, Serialize)]
pub struct OwnedBenchmarkGroupData {
    /// The name of a chart and its associated data.
    pub charts: HashMap<String, OwnedChartData>,
}

/// Chart data with owned strings.
#[derive(Debug, Clone, Serialize)]
pub struct OwnedChartData {
    /// The name of a series and its associated data.
    pub aligned_series: HashMap<String, Vec<Option<u64>>>,
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
pub fn process_benchmarks<'a>(
    entries: &'a [BenchmarkEntry],
    sorted_commits: &[CommitInfo],
) -> VortexResult<Benchmarks<'a>> {
    log!(
        "[process_benchmarks] Starting with {} entries, {} commits",
        entries.len(),
        sorted_commits.len()
    );

    if !sorted_commits.is_sorted() {
        vortex_bail!("Commits must be sorted by timestamp");
    }

    let num_commits = sorted_commits.len();
    let grouped_entries = BenchmarkEntry::group(entries);

    log!(
        "[process_benchmarks] Grouped into {} groups: {:?}",
        grouped_entries.len(),
        grouped_entries.keys().collect::<Vec<_>>()
    );

    let mut benchmarks = HashMap::with_capacity(grouped_entries.keys().len());
    for (group_name, group_data) in grouped_entries {
        if group_name.is_empty() {
            vortex_bail!("Group name cannot be empty");
        }

        log!(
            "[process_benchmarks] Group '{}' has {} charts",
            group_name,
            group_data.len()
        );

        let mut charts = HashMap::with_capacity(group_data.keys().len());
        for (chart_name, chart_data) in group_data {
            if chart_name.is_empty() {
                vortex_bail!("Chart name cannot be empty in group '{}'", group_name);
            }

            let mut aligned_series = HashMap::with_capacity(chart_data.keys().len());
            for (series_name, series_data) in chart_data {
                if series_name.is_empty() {
                    vortex_bail!(
                        "Series name cannot be empty in group '{}', chart '{}'",
                        group_name,
                        chart_name
                    );
                }

                let aligned_series_data = create_aligned_series_data(series_data, sorted_commits);

                // Validate series length matches commits.
                if aligned_series_data.len() != num_commits {
                    vortex_bail!(
                        "Series '{}' in group '{}', chart '{}' has {} elements, expected {} (number of commits)",
                        series_name,
                        group_name,
                        chart_name,
                        aligned_series_data.len(),
                        num_commits
                    );
                }

                // Skip series with no data points (all nulls).
                let data_points = aligned_series_data.iter().filter(|v| v.is_some()).count();
                if data_points == 0 {
                    log!(
                        "[process_benchmarks]   Pruning series '{}' (no data points)",
                        series_name
                    );
                    continue;
                }

                aligned_series.insert(series_name, aligned_series_data);
            }

            // Skip charts with no series.
            if aligned_series.is_empty() {
                log!(
                    "[process_benchmarks]   Pruning chart '{}' (no series with data)",
                    chart_name
                );
                continue;
            }

            log!(
                "[process_benchmarks]   Chart '{}' has {} series with data",
                chart_name,
                aligned_series.len()
            );

            let chart_data = ChartData { aligned_series };
            charts.insert(chart_name, chart_data);
        }

        // Skip groups with no charts.
        if charts.is_empty() {
            log!(
                "[process_benchmarks] Pruning group '{}' (no charts with data)",
                group_name
            );
            continue;
        }

        log!(
            "[process_benchmarks] Group '{}' retained {} charts",
            group_name,
            charts.len()
        );

        let benchmark_group_data = BenchmarkGroupData { charts };
        benchmarks.insert(group_name, benchmark_group_data);
    }

    log!(
        "[process_benchmarks] Done. Final groups: {:?}",
        benchmarks.keys().collect::<Vec<_>>()
    );

    Ok(benchmarks)
}

fn create_aligned_series_data(
    commits_and_values: CommitValueMap<'_>,
    sorted_commits: &[CommitInfo],
) -> Vec<Option<u64>> {
    sorted_commits
        .iter()
        .map(|commit_info| commits_and_values.get(commit_info.commit_id()).copied())
        .collect()
}

// ============================================================================
// Owned processing functions
// ============================================================================

/// Processes benchmark entries into owned structures suitable for static caching.
///
/// This is similar to [`process_benchmarks`] but returns owned data that can be stored
/// in a `OnceLock` for caching.
pub fn process_benchmarks_owned(
    entries: &[BenchmarkEntry],
    sorted_commits: &[CommitInfo],
) -> VortexResult<OwnedBenchmarks> {
    log!(
        "[process_benchmarks_owned] Starting with {} entries, {} commits",
        entries.len(),
        sorted_commits.len()
    );

    if !sorted_commits.is_sorted() {
        vortex_bail!("Commits must be sorted by timestamp");
    }

    let num_commits = sorted_commits.len();
    let grouped_entries = BenchmarkEntry::group(entries);

    log!(
        "[process_benchmarks_owned] Grouped into {} groups",
        grouped_entries.len()
    );

    let mut benchmarks = HashMap::with_capacity(grouped_entries.keys().len());
    for (group_name, group_data) in grouped_entries {
        if group_name.is_empty() {
            vortex_bail!("Group name cannot be empty");
        }

        let mut charts = HashMap::with_capacity(group_data.keys().len());
        for (chart_name, chart_data) in group_data {
            if chart_name.is_empty() {
                vortex_bail!("Chart name cannot be empty in group '{}'", group_name);
            }

            let mut aligned_series = HashMap::with_capacity(chart_data.keys().len());
            for (series_name, series_data) in chart_data {
                if series_name.is_empty() {
                    vortex_bail!(
                        "Series name cannot be empty in group '{}', chart '{}'",
                        group_name,
                        chart_name
                    );
                }

                let aligned_series_data = create_aligned_series_data(series_data, sorted_commits);

                if aligned_series_data.len() != num_commits {
                    vortex_bail!(
                        "Series '{}' in group '{}', chart '{}' has {} elements, expected {}",
                        series_name,
                        group_name,
                        chart_name,
                        aligned_series_data.len(),
                        num_commits
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

            charts.insert(chart_name.to_string(), OwnedChartData { aligned_series });
        }

        if charts.is_empty() {
            continue;
        }

        benchmarks.insert(group_name.to_string(), OwnedBenchmarkGroupData { charts });
    }

    log!(
        "[process_benchmarks_owned] Done. Final groups: {:?}",
        benchmarks.keys().collect::<Vec<_>>()
    );

    Ok(benchmarks)
}

/// Extracts summary metadata from owned benchmarks.
pub fn extract_summary(benchmarks: &OwnedBenchmarks, commits: Vec<CommitInfo>) -> BenchmarkSummary {
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
