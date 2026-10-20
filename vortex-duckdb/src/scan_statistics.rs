// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Statistics of the columns of a scan over several files, merged from the footers of the files.
//!
//! DuckDB only reports statistics of a scan over several files when it reads them by name, as it
//! would have to read the metadata of every file. Vortex scans cache the footers of the files they
//! open, so once every file of a scan was opened, the statistics of its columns are merged from
//! the footers. DuckDB then plans with them as it plans its own tables: it drops overflow checks
//! from arithmetic, prunes filters, and packs group keys.

use std::ptr;
use std::sync::Arc;
use std::sync::LazyLock;

use parking_lot::Mutex;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::dtype::StructFields;
use vortex::error::VortexExpect;
use vortex::error::vortex_panic;
use vortex::file::FileStatistics;
use vortex::file::multi::MultiFileSession;
use vortex::layout::layouts::file_stats::AggregateStat;
use vortex::layout::layouts::file_stats::AggregateStats;
use vortex::scalar::Scalar;
use vortex::session::SessionExt as _;
use vortex_utils::aliases::hash_map::HashMap;

use crate::SESSION;
use crate::column_statistics::ColumnStatistics;
use crate::column_statistics::ColumnStatisticsAggregate;
use crate::file_reader::cache_local_footer;

/// The statistics of each column of a set of files.
pub struct MergedStatistics {
    fields: StructFields,
    stats: Vec<StatsSet>,
    max_lengths: Vec<Option<u64>>,
}

impl MergedStatistics {
    /// The statistics of the column called `name`.
    ///
    /// The distinct count of each file says little about the count of all files, so it is left
    /// unknown.
    pub fn column(&self, name: &str) -> Option<ColumnStatistics> {
        let index = self.fields.find(name)?;
        let dtype = self.fields.field_by_index(index)?;
        let aggregates = aggregates_from_stats_set(&self.stats[index], &dtype);
        let stats = ColumnStatisticsAggregate::new(&aggregates, None, self.max_lengths[index]);
        match ColumnStatistics::try_from(stats, dtype) {
            Ok(stats) => Some(stats),
            Err(e) => vortex_panic!(e),
        }
    }
}

/// The aggregates of a column of `dtype` that `stats` holds, for the stats that have an aggregate
/// function.
fn aggregates_from_stats_set(stats: &StatsSet, dtype: &DType) -> AggregateStats {
    AggregateStats::new(
        stats
            .iter()
            .filter_map(|(stat, value)| {
                let aggregate_fn = stat.aggregate_fn()?;
                let stat_dtype = stat.dtype(dtype)?;
                let value = value
                    .clone()
                    .map(|value| Scalar::try_new(stat_dtype.clone(), Some(value)))
                    .transpose()
                    .ok()?;
                Some(AggregateStat::from_value(aggregate_fn, value))
            })
            .collect(),
    )
}

/// The statistics of the column at `path` of a file, or none if the file has no statistics of
/// it.
fn column_stats_set(file: &FileStatistics, path: &FieldPath) -> StatsSet {
    file.get_by_path(path)
        .map(|(aggregates, _)| aggregates.to_stats_set())
        .unwrap_or_default()
}

/// Merged statistics with the file statistics they were merged from.
struct CachedStatistics {
    /// Held so that their identity tells whether a footer was replaced since.
    files: Vec<FileStatistics>,
    merged: Arc<MergedStatistics>,
}

/// The number of file sets whose merged statistics are kept.
const MAX_CACHED_FILE_SETS: usize = 64;

/// Merged statistics by the paths of the files they were merged from.
///
/// DuckDB asks for the statistics of every column of a scan while it orders joins, before it
/// removes unused columns, so merging them for every query would cost a pass over all the columns
/// of all the files.
static CACHE: LazyLock<Mutex<HashMap<Vec<String>, CachedStatistics>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The statistics of the columns of the files at `paths`, if the footers of all of them are
/// cached or local, and have statistics.
///
/// The footers of local files that no scan has opened yet are read here, so that the first query
/// over a set of files plans with their statistics too. Reading the footers of remote files would
/// cost requests that planning should not make.
pub fn merged_statistics(paths: &[&str]) -> Option<Arc<MergedStatistics>> {
    let session = SESSION
        .get_opt::<MultiFileSession>()
        .vortex_expect("MultiFileSession not found");
    let footers = paths
        .iter()
        .map(|path| {
            session.get_footer(path).or_else(|| {
                // A file that fails to open plans without statistics, and its scan reports why.
                cache_local_footer(path).ok()?;
                session.get_footer(path)
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let files = footers
        .iter()
        .map(|footer| footer.statistics().cloned())
        .collect::<Option<Vec<_>>>()?;

    let key: Vec<String> = paths.iter().map(|path| (*path).to_owned()).collect();
    let mut cache = CACHE.lock();
    if let Some(cached) = cache.get(&key)
        && cached.files.len() == files.len()
        && cached
            .files
            .iter()
            .zip(&files)
            .all(|(cached, file)| ptr::eq(cached.aggregates().as_ptr(), file.aggregates().as_ptr()))
    {
        return Some(Arc::clone(&cached.merged));
    }

    let (first, rest) = footers.split_first()?;
    let fields = first.dtype().as_struct_fields_opt()?.clone();
    if rest.iter().any(|footer| footer.dtype() != first.dtype()) {
        return None;
    }
    let dtypes: Vec<_> = fields.fields().collect();
    let paths: Vec<_> = fields
        .names()
        .iter()
        .map(|name| FieldPath::from_name(name.clone()))
        .collect();
    let (first_file, rest_files) = files.split_first()?;
    let mut stats: Vec<StatsSet> = paths
        .iter()
        .map(|path| column_stats_set(first_file, path))
        .collect();
    let mut max_lengths: Vec<Option<u64>> = paths
        .iter()
        .map(|path| first_file.max_length_by_path(path))
        .collect();
    for file in rest_files {
        for (index, (dtype, path)) in dtypes.iter().zip(&paths).enumerate() {
            let merged = std::mem::take(&mut stats[index]);
            stats[index] = merged.merge_unordered(&column_stats_set(file, path), dtype);
            max_lengths[index] = max_lengths[index]
                .zip(file.max_length_by_path(path))
                .map(|(a, b)| a.max(b));
        }
    }
    let merged = Arc::new(MergedStatistics {
        fields,
        stats,
        max_lengths,
    });

    if cache.len() >= MAX_CACHED_FILE_SETS {
        cache.clear();
    }
    cache.insert(
        key,
        CachedStatistics {
            files,
            merged: Arc::clone(&merged),
        },
    );
    Some(merged)
}
