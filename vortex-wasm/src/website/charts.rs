// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use serde::Deserialize;
use serde::Serialize;
use vortex::utils::aliases::hash_map::HashMap;

use crate::website::commit::CommitInfo;
use crate::website::entry::BenchmarkEntry;
use crate::website::entry::CommitValueMap;

type Benchmarks<'a> = HashMap<&'a str, BenchmarkGroupData<'a>>;

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

pub fn process_benchmarks<'a>(
    entries: &'a [BenchmarkEntry],
    sorted_commits: &[CommitInfo],
) -> Benchmarks<'a> {
    assert!(sorted_commits.is_sorted());

    let grouped_entries = BenchmarkEntry::group(entries);

    let mut benchmarks = HashMap::with_capacity(grouped_entries.keys().len());
    for (group_name, group_data) in grouped_entries {
        let mut charts = HashMap::with_capacity(group_data.keys().len());
        for (chart_name, chart_data) in group_data {
            let mut aligned_series = HashMap::with_capacity(chart_data.keys().len());
            for (series_name, series_data) in chart_data {
                let aligned_series_data = create_aligned_series_data(series_data, sorted_commits);
                aligned_series.insert(series_name, aligned_series_data);
            }

            let chart_data = ChartData { aligned_series };
            charts.insert(chart_name, chart_data);
        }

        let benchmark_group_data = BenchmarkGroupData { charts };
        benchmarks.insert(group_name, benchmark_group_data);
    }

    benchmarks
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
