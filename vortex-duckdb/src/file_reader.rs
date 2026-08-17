// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use object_store::registry::ObjectStoreRegistry;
use tracing::debug;
use url::Url;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::cloud::Registry;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
use vortex::expr::BoundExpression;
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
use vortex::scan::selection::Selection;

use crate::RUNTIME;
use crate::SESSION;
use crate::column_statistics::ColumnStatistics;
use crate::column_statistics::ColumnStatisticsAggregate;
use crate::cpp;
use crate::duckdb::DataChunkRef;
use crate::duckdb::TableFilterSet;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::projection::FILE_INDEX_COLUMN_IDX;
use crate::projection::FILE_ROW_NUMBER_COLUMN_IDX;
use crate::projection::Filter;
use crate::table_function::Split;
use crate::table_function::TableFunctionBind;
use crate::table_function::TableFunctionGlobal;
use crate::table_function::TableFunctionLocal;
use crate::table_function::convert_result;
use crate::table_function::optimize_and_bind;

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
    /// File splits stored in inverse order.
    pub splits: Vec<Split>,
    pub cache: ConversionCache,
    file_index: u64,
    total_splits: u64,
}

async fn open_reader(file_path: String, file_index: u64) -> VortexResult<File> {
    let url = parse_uri_or_path(&file_path)?;
    let (fs, path) = resolve_filesystem(&url)?;
    let file = fs.open_read(&path).await?;
    let file = open_cached(&SESSION, file, &path, None, &|options| options).await?;
    Ok(File {
        reader: file.layout_reader()?,
        file_index,
        total_splits: 0,
        cache: ConversionCache::default(),
        splits: vec![],
        file_row_number_column_pos: None,
        file_index_column_pos: None,
    })
}

pub fn reader_open(file_path: &str, file_index: u64) -> VortexResult<File> {
    RUNTIME.block_on(open_reader(file_path.to_owned(), file_index))
}

fn reader_prune(global: &TableFunctionGlobal, file: &File) -> VortexResult<bool> {
    let Some(pruning) = global.pruning.as_ref() else {
        return Ok(false);
    };
    let index = file.file_index;
    let excluded = match &pruning.file_selection {
        Selection::IncludeByIndex(buffer) => buffer.as_slice().binary_search(&index).is_err(),
        Selection::ExcludeByIndex(buffer) => buffer.as_slice().binary_search(&index).is_ok(),
        _ => false,
    };
    if excluded
        || pruning
            .file_range
            .as_ref()
            .is_some_and(|r| !r.contains(&index))
    {
        return Ok(true);
    }

    let Some(filter) = &pruning.filter else {
        return Ok(false);
    };
    let row_count = file.reader.row_count();
    let row_range = 0..row_count;
    let mask = Mask::new_true(usize::try_from(row_count).unwrap_or(usize::MAX));
    let evaluation = file.reader.pruning_evaluation(&row_range, filter, mask)?;
    match evaluation.now_or_never() {
        Some(Ok(result_mask)) => Ok(result_mask.all_false()),
        _ => Ok(false),
    }
}

/// Returns true if file should be skipped.
/// Called under file lock.
pub fn reader_initialize(global: &TableFunctionGlobal, file: &File) -> VortexResult<bool> {
    if reader_prune(global, file)? {
        return Ok(true);
    }

    for (i, id) in column_ids.iter().enumerate() {
        if *id == FILE_ROW_NUMBER_COLUMN_IDX {
            file.file_row_number_column_pos = Some(i);
        } else if *id == FILE_INDEX_COLUMN_IDX {
            file.file_index_column_pos = Some(i);
        }
    }

    let Filter {
        filter,
        row_selection,
        row_range,
        has_non_optional_filter,
        file_selection,
        file_range,
    } = convert_filter(bind, column_ids, filters)?;
    if has_non_optional_filter {
        bind.has_non_optional_filter.store(true, Ordering::Relaxed);
    }

    debug!(
        filter = filter
            .as_ref()
            .map_or_else(|| "true".to_string(), |f| f.to_string()),
        ?row_selection,
        ?row_range,
        ?file_selection,
        ?file_range,
        "prepare scan"
    );

    let filter = filter
        .map(|expr| optimize_and_bind(expr, &bind.dtype))
        .transpose()?;

    let mut builder = ScanBuilder::new(SESSION.clone(), Arc::clone(&file.reader))
        .with_projection(global.projection.clone())
        .with_some_filter(filter)
        .with_ordered(file_row_number_column_pos.is_some())
        .with_selection(row_selection);
    if let Some(range) = row_range {
        builder = builder.with_row_range(range);
    }
    let splits = builder.build()?;

    let handles = splits
        .into_iter()
        .map(|task| RUNTIME.handle().spawn(task))
        .collect::<Vec<_>>();
}

/// Returns true if file is exhausted.
/// Called under global lock.
pub fn reader_try_initialize_scan(
    bind: &TableFunctionBind,
    global: &TableFunctionGlobal,
    local: &mut TableFunctionLocal,
    file: &mut File,
    column_ids: &[u64],
    filters: cpp::duckdb_vx_table_filter_set,
) -> VortexResult<bool> {
    if let Some(split) = file.splits.pop() {
        local.split = Some(split);
        local.file_row_number_column_pos = file.file_row_number_column_pos;
        local.file_index_column_pos = file.file_index_column_pos;
        return Ok(false);
    }
    if file.total_splits > 0 {
        // file is exhausted
        return Ok(true);
    }

    let pending = (!global.aggregates.is_empty()).then(|| Arc::clone(&global.pending));
    // TODO we may want some backpressure/size (number of cores?)
    let (sender, receiver) = kanal::unbounded_async();

    let driver = RUNTIME.handle().spawn(async move {
        for handle in handles {
            match handle.await {
                Ok(Some(array)) => {
                    if let Some(pending) = &pending {
                        pending.fetch_add(1, Ordering::Relaxed);
                    }
                    if sender.send(Ok(array)).await.is_err() {
                        // The receiver is gone: the scan was cancelled or the
                        // query ended.
                        return;
                    }
                }
                // split is filtered
                Ok(None) => {}
                Err(e) => {
                    let _ = sender.send(Err(e)).await;
                    return;
                }
            }
        }
    });

    local.file_row_number_column_pos = file_row_number_column_pos;
    local.file_index_column_pos = file_index_column_pos;
    file.driver = Some(Driver { driver, receiver });
    Ok(false)
}

pub fn file_statistics(file: &File, column: &str) -> Option<ColumnStatistics> {
    let reader = file
        .reader
        .as_any()
        .downcast_ref::<FileStatsLayoutReader>()?;
    let stats_sets = reader.file_stats().stats_sets();

    let DType::Struct(fields, _) = &file.reader.dtype() else {
        vortex_panic!("Not a Struct");
    };
    let index = fields.find(column)?;
    let dtype = fields.field_by_index(index)?;

    let stats = ColumnStatisticsAggregate::new(stats_sets.get(index)?);
    Some(ColumnStatistics::from(&stats, dtype))
}

pub struct Pruning {
    filter: Option<BoundExpression>,
    file_selection: Selection,
    file_range: Option<Range<u64>>,
}

fn convert_filter(
    bind: &TableFunctionBind,
    column_ids: &[u64],
    filters: cpp::duckdb_vx_table_filter_set,
) -> VortexResult<Filter> {
    let table_filter_set = if filters.is_null() {
        None
    } else {
        Some(unsafe { TableFilterSet::borrow(filters) })
    };
    Filter::new(
        table_filter_set,
        column_ids,
        bind.column_fields.as_slice(),
        &bind.filter_exprs,
        &bind.dtype,
    )
}

impl Pruning {
    pub fn new(
        bind: &TableFunctionBind,
        column_ids: &[u64],
        filters: cpp::duckdb_vx_table_filter_set,
    ) -> VortexResult<Option<Self>> {
        if filters.is_null() && bind.filter_exprs.is_empty() {
            return Ok(None);
        }
        let converted = convert_filter(bind, column_ids, filters)?;
        let filter = converted
            .filter
            .map(|expr| optimize_and_bind(expr, &bind.dtype))
            .transpose()?;
        if filter.is_none()
            && matches!(converted.file_selection, Selection::All)
            && converted.file_range.is_none()
        {
            return Ok(None);
        }
        Ok(Some(Self {
            filter,
            file_selection: converted.file_selection,
            file_range: converted.file_range,
        }))
    }
}

fn file_scan_aggregate(
    file: &mut File,
    global: &TableFunctionGlobal,
    local: &mut TableFunctionLocal,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let has_count_star = local.partials.len() < global.aggregates.len();
    let mut accumulated = 0u64;
    let mut rows = 0u64;

    let Some(receiver) = local.receiver else {
        vortex_panic!("no receiver");
    };

    loop {
        let Ok(array) = RUNTIME.block_on(receiver.recv()) else {
            break;
        };
        let array = convert_result(array?, &mut ctx)?;

        for (position, partial) in global
            .aggregate_positions
            .iter()
            .zip(local.partials.iter_mut())
        {
            partial.accumulate(array.unmasked_field(*position), &mut ctx)?;
        }

        let len = array.len() as u64;
        rows += len;
        file.rows_read.fetch_add(len, Ordering::Relaxed);
        accumulated += 1;
    }

    if accumulated == 0 {
        return Ok(());
    }

    {
        let mut partials = global.partials.lock();
        for (global_partial, local_partial) in partials.iter_mut().zip(&mut local.partials) {
            global_partial.combine_partials(local_partial.flush()?)?;
        }
    }
    if has_count_star {
        global.row_count.fetch_add(rows, Ordering::Relaxed);
    }
    global.pending.fetch_sub(accumulated, Ordering::Release);
    Ok(())
}

/// Returns true if file is exhausted
pub fn file_scan(
    file: &File,
    global: &TableFunctionGlobal,
    local: &mut TableFunctionLocal,
    output: &mut DataChunkRef,
) -> VortexResult<bool> {
    if !local.partials.is_empty() {
        file_scan_aggregate(file, global, local)?;
        return Ok(true);
    }
    let Some(scan) = file.scan.as_ref() else {
        vortex_panic!("No file scan");
    };
    loop {
        if local.exporter.is_none() {
            let Ok(item) = RUNTIME.block_on(scan.receiver.recv()) else {
                return Ok(true);
            };
            let (array, cache) = item?;

            let mut ctx = SESSION.create_execution_ctx();
            let array = convert_result(array, &mut ctx)?;
            file.rows_read
                .fetch_add(array.len() as u64, Ordering::Relaxed);
            local.exporter = Some(ArrayExporter::try_new(&array, &cache, ctx)?);
        }

        let exporter = local
            .exporter
            .as_mut()
            .ok_or_else(|| vortex_err!("exporter missing"))?;
        let has_more_data = exporter.export(
            output,
            scan.file_index_column_pos,
            scan.file_row_number_column_pos,
        )?;

        if !has_more_data {
            // This exporter is fully consumed.
            local.exporter = None;
        } else {
            break;
        }
    }
    Ok(false)
}

pub fn get_progress_in_file(file: &File) -> f64 {
    // TODO(myrrc) this is inaccurate if filters are pushed
    let read = file.rows_read.load(Ordering::Relaxed) as f64;
    100.0 * read / file.reader.row_count() as f64
}
