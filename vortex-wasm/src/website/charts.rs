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

/// A map of group names to their benchmark data.
pub type Benchmarks<'a> = HashMap<&'a str, BenchmarkGroupData<'a>>;

/// The complete response containing benchmarks and commit metadata.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResponse<'a> {
    /// Benchmarks grouped by group name, chart name, and series name.
    pub benchmarks: Benchmarks<'a>,
    /// Sorted list of commits (by timestamp). Owned to allow serialization.
    pub commits: Vec<CommitInfo>,
}

// TODO(connor): We should be able to use an `Option<NonZeroU64>` since our benchmarks should
// basically never hit 0, but that is an optimization for another day.
type AlignedSeries<'a> = HashMap<&'a str, Vec<Option<u64>>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkGroupData<'a> {
    /// The name of a chart and its associated data.
    #[serde(borrow)]
    charts: HashMap<&'a str, ChartData<'a>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData<'a> {
    /// The name of a series and its associated data.
    #[serde(borrow)]
    aligned_series: AlignedSeries<'a>,
}

/// Processes benchmark entries into a structured format aligned with commits.
///
/// # Errors
///
/// Returns an error if:
/// - Commits are not sorted by timestamp.
/// - Any group, chart, or series name is empty.
/// - Any series has no data points (all nulls).
/// - Series lengths don't match the number of commits.
pub fn process_benchmarks<'a>(
    entries: &'a [BenchmarkEntry],
    sorted_commits: &[CommitInfo],
) -> VortexResult<Benchmarks<'a>> {
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

                // Validate series has at least one non-null value.
                if !aligned_series_data.iter().any(|v| v.is_some()) {
                    vortex_bail!(
                        "Series '{}' in group '{}', chart '{}' has no data points (all nulls)",
                        series_name,
                        group_name,
                        chart_name
                    );
                }

                aligned_series.insert(series_name, aligned_series_data);
            }

            let chart_data = ChartData { aligned_series };
            charts.insert(chart_name, chart_data);
        }

        let benchmark_group_data = BenchmarkGroupData { charts };
        benchmarks.insert(group_name, benchmark_group_data);
    }

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
