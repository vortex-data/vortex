// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::cloud::Registry;
use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_panic;
use vortex::file::multi::open_cached;
use vortex::file::multi::parse_uri_or_path;
use vortex::file::v2::FileStatsLayoutReader;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::layout::LayoutReaderRef;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::mask::Mask;

use crate::RUNTIME;
use crate::SESSION;
use crate::column_statistics::ColumnStatistics;
use crate::column_statistics::ColumnStatisticsAggregate;
use crate::duckdb::BindResultRef;
use crate::duckdb::DataChunkRef;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::projection::Filter;
use crate::projection::extract_schema_from_dtype;
use crate::table_function::BindState;
use crate::table_function::GlobalState;
use crate::table_function::LocalState;
use crate::table_function::Split;
use crate::table_function::convert_result;

/// Duckdb invokes following callbacks under various locks, and information
/// about these locks is not propagated to Rust. We, however, don't want to add
/// excessive synchronisation on our side, so the best we can do is document
/// the invariants here.
///
/// Definitions:
/// "global lock" - lock over all threads. Only one thread may access some
///   section.
/// "file-local lock" - multiple threads may access different File's in
///   parallel, but only one thread can access a single File at a time.
///
/// Lifetime of file:
///
/// Bind, first file:
///
/// reader_open -> reader_bind
///
/// Plan and run:
///
/// reader_open (plan time) -> reader_get_statistics (plan time) ->
/// reader_initialize -> reader_try_initialize_scan -> reader_scan
///
/// reader_get_progress_in_file is called between
/// reader_scan calls from a separate thread.

static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

fn resolve_filesystem(url: &Url) -> VortexResult<(FileSystemRef, String)> {
    // Compat makes us use tokio which is very bad for local reads on
    // high-core machines because reads go into blocking pool
    if url.scheme() == "file" {
        return Ok((
            Arc::new(ObjectStoreFileSystem::local(RUNTIME.handle())),
            url.path().to_string(),
        ));
    }

    let (object_store, path) = REGISTRY.resolve(url)?;

    Ok((
        Arc::new(ObjectStoreFileSystem::new(
            Arc::new(Compat::new(object_store)),
            RUNTIME.handle(),
        )),
        path.to_string(),
    ))
}

pub struct File {
    pub reader: LayoutReaderRef,
    /// File splits stored in inverse order
    pub splits: Vec<Split>,
    pub cache: ConversionCache,
    total_splits: usize,
}

impl File {
    async fn open(file_path: String) -> VortexResult<Self> {
        let url = parse_uri_or_path(&file_path)?;
        let (fs, path) = resolve_filesystem(&url)?;
        let file = fs.open_read(&path).await?;
        let file = open_cached(&SESSION, file, &path, None, &|options| options).await?;
        Ok(File {
            reader: file.layout_reader()?,
            cache: ConversionCache::default(),
            splits: vec![],
            total_splits: 0,
        })
    }

    fn can_skip(&self, filter: &Filter) -> VortexResult<bool> {
        let Some(filter) = &filter.filter else {
            return Ok(false);
        };
        let row_count = self.reader.row_count();
        let row_range = 0..row_count;
        let mask = Mask::new_true(usize::try_from(row_count).unwrap_or(usize::MAX));
        let evaluation = self.reader.pruning_evaluation(&row_range, filter, mask)?;
        match evaluation.now_or_never() {
            Some(mask) => mask.map(|mask| mask.all_false()),
            None => Ok(false),
        }
    }
}

/// Called once per file while initializing the scan under file-local lock.
/// Files are opened lazily.
pub fn reader_open(file_path: &str) -> VortexResult<File> {
    RUNTIME.block_on(File::open(file_path.to_owned()))
}

/// Called once per scan with first file without locks. Populates "result"
/// with first file schema which is the scan schema. Unlike Parquet, we don't
/// support schema evolution, so if any file schema doesn't match first schema,
/// we break.
pub fn reader_bind(file: &File, result: &mut BindResultRef) -> VortexResult<BindState> {
    let dtype = file.reader.dtype().clone();
    let columns = extract_schema_from_dtype(&dtype)?;

    for column in &columns {
        result.add_result_column(&column.name, &column.logical_type);
    }

    Ok(BindState {
        dtype,
        first_file_row_count: file.reader.row_count(),
        filters: vec![],
        columns,
        has_non_optional_filter: AtomicBool::new(false),
        aggregates: vec![],
    })
}

/// Called once per file by one thread under file-local lock. Determines
/// whether the opened file should be skipped. If this function returns false,
/// duckdb closes the file and doesn't call reader_try_initialize_scan on it.
pub fn reader_initialize(file: &mut File, global: &GlobalState) -> VortexResult<bool> {
    if file.can_skip(&global.filter)? {
        return Ok(true);
    }

    // Getting splits is non-trivial work so we prefer doing it here under file
    // lock and not in reader_try_initialize_scan under global lock.
    let ordered = global.file_row_number_column_pos.is_some();
    let reader = Arc::clone(&file.reader);
    let filter = &global.filter;
    let mut builder = ScanBuilder::new(SESSION.clone(), reader)
        .with_projection(global.projection.clone())
        .with_ordered(ordered)
        .with_some_filter(filter.filter.clone())
        .with_selection(filter.row_selection.clone());
    if let Some(row_range) = filter.row_range.as_ref() {
        builder = builder.with_row_range(row_range.clone());
    }
    let mut splits = builder.build()?;

    // threads take last element of file.splits so we need to reverse
    splits.reverse();
    file.total_splits = splits.len();
    file.splits = splits;
    Ok(false)
}

/// Called by all threads under global lock. If this function returns true,
/// thread calls reader_scan on this file. If this function returns false,
/// duckdb thinks file is exhausted, closes the file, and the first thread to
/// get "false" switches to next file.
pub fn reader_try_initialize_scan(file: &mut File, local: &mut LocalState) -> bool {
    let Some(split) = file.splits.pop() else {
        return false;
    };
    local.split = Some(split);
    true
}

/// Called by all threads operating on a file without locks. If this function
/// returns false, duckdb closes the file, and first thread to get "false"
/// switches to next file.
pub fn reader_scan(
    file: &File,
    global: &GlobalState,
    local: &mut LocalState,
    chunk: &mut DataChunkRef,
) -> VortexResult<bool> {
    if !global.aggregates.is_empty() {
        return reader_scan_aggregate(global, local);
    }

    if local.exporter.is_none() {
        let Some(split) = local.split.take() else {
            return Ok(false);
        };
        let Some(array) = RUNTIME.block_on(split)? else {
            // split is filtered
            return Ok(true);
        };
        let mut ctx = SESSION.create_execution_ctx();
        let array = convert_result(array, &mut ctx)?;
        local.exporter = Some(ArrayExporter::try_new(&array, &file.cache, ctx)?);
    }
    let exporter = local.exporter.as_mut().vortex_expect("no exporter");

    let has_more_data = exporter.export(chunk, global.file_row_number_column_pos)?;
    if !has_more_data {
        local.exporter = None;
    }
    Ok(true)
}

fn reader_scan_aggregate(global: &GlobalState, local: &mut LocalState) -> VortexResult<bool> {
    let Some(split) = local.split.take() else {
        return Ok(false);
    };
    let Some(array) = RUNTIME.block_on(split)? else {
        // split is filtered
        return Ok(true);
    };

    let mut ctx = SESSION.create_execution_ctx();
    let array = convert_result(array, &mut ctx)?;

    for (position, partial) in local.partials.iter_mut() {
        partial.accumulate(array.unmasked_field(*position), &mut ctx)?;
    }

    if global.has_count_star {
        let len = array.len() as u64;
        global.row_count.fetch_add(len, Ordering::Relaxed);
    }

    Ok(true)
}

/// Called by one thread in plan phase without locks
pub fn reader_get_statistics(file: &File, column: &str) -> Option<ColumnStatistics> {
    let reader = file
        .reader
        .as_any()
        .downcast_ref::<FileStatsLayoutReader>()?;
    let stats_sets = reader.file_stats().stats_sets();

    let DType::Struct(fields, _) = &file.reader.dtype() else {
        return None;
    };
    let index = fields.find(column)?;
    let dtype = fields.field_by_index(index)?;

    let stats = ColumnStatisticsAggregate::new(stats_sets.get(index)?);
    match ColumnStatistics::from(&stats, dtype) {
        Ok(stats) => Some(stats),
        Err(e) => vortex_panic!(e),
    }
}

/// Called from a separate thread (not related to threads for Vortex
/// table function) under global lock.
pub fn reader_get_progress_in_file(file: &File) -> f64 {
    let total = file.total_splits;
    let left = file.splits.len();
    let denom = total + (total == 0) as usize;
    100.0 * (total - left) as f64 / denom as f64
}
