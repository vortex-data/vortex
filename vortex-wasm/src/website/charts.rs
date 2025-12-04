// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use serde::Deserialize;
use serde::Serialize;
use vortex::utils::aliases::hash_map::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkGroup<'a> {
    group_name: &'a str,
    #[serde(borrow)]
    charts: HashMap<&'a str, ChartData<'a>>,
}

// TODO(connor): We should be able to use an `Option<NonZeroU64>` since our benchmarks should
// basically never hit 0, but that is an optimization for another day.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData<'a> {
    commit_index_range: Range<usize>,
    #[serde(borrow)]
    series: HashMap<&'a str, Vec<Option<u64>>>,
}
