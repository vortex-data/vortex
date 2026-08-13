// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use kanal::AsyncReceiver;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::cloud::Registry;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::BoundExpression;
use vortex::file::multi::open_cached;
use vortex::file::multi::parse_uri_or_path;
use vortex::file::v2::FileStatsLayoutReader;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::io::runtime::Task;
use vortex::layout::LayoutReaderRef;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::mask::Mask;
use vortex::scan::selection::Selection;

use crate::RUNTIME;
use crate::SESSION;
use crate::column_statistics::ColumnStatistics;
use crate::column_statistics::ColumnStatisticsAggregate;
use crate::cpp;
use crate::duckdb::BindResultRef;
use crate::duckdb::DataChunkRef;
use crate::duckdb::TableFilterSet;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::projection::FILE_INDEX_COLUMN_IDX;
use crate::projection::FILE_ROW_NUMBER_COLUMN_IDX;
use crate::projection::Filter;
use crate::projection::extract_schema_from_dtype;
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

type ScanItem = VortexResult<(ArrayRef, Arc<ConversionCache>)>;

pub struct FileReader {
    reader: LayoutReaderRef,
    pub(crate) dtype: DType,
    pub(crate) row_count: u64,
    file_index: u64,
}

pub struct FileScan {
    receiver: AsyncReceiver<ScanItem>,
    total_splits: u64,
    delivered: AtomicU64,
    file_row_number_column_pos: Option<usize>,
    file_index_column_pos: Option<usize>,
    aggregate_positions: Vec<usize>,
    _driver: Task<()>,
}

async fn open_reader(file_path: String, file_index: u64) -> VortexResult<FileReader> {
    let url = parse_uri_or_path(&file_path)?;
    let (fs, path) = resolve_filesystem(&url)?;
    let source = fs.open_read(&path).await?;
    let vortex_file = open_cached(&SESSION, source, &path, None, &|options| options).await?;
    let reader = vortex_file.layout_reader()?;
    Ok(FileReader {
        dtype: reader.dtype().clone(),
        row_count: reader.row_count(),
        reader,
        file_index,
    })
}

pub fn file_open(file_path: &str, file_index: u64) -> VortexResult<FileReader> {
    RUNTIME.block_on(open_reader(file_path.to_owned(), file_index))
}

pub fn file_schema(file: &FileReader, result: &mut BindResultRef) -> VortexResult<()> {
    for field in extract_schema_from_dtype(&file.dtype)? {
        result.add_result_column(&field.name, &field.logical_type);
    }
    Ok(())
}

pub fn file_statistics(file: &FileReader, column_name: &str) -> Option<ColumnStatistics> {
    let stats_reader = file
        .reader
        .as_any()
        .downcast_ref::<FileStatsLayoutReader>()?;
    let stats_sets = stats_reader.file_stats().stats_sets();

    let DType::Struct(fields, _) = &file.dtype else {
        return None;
    };
    let index = fields
        .names()
        .iter()
        .position(|name| name.as_ref() == column_name)?;
    let dtype = fields.field_by_index(index)?;

    let stats_aggregate = ColumnStatisticsAggregate::new(stats_sets.get(index)?);
    Some(ColumnStatistics::from(&stats_aggregate, dtype))
}

pub struct ScanPruning {
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

impl ScanPruning {
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

pub fn file_should_skip(global: &TableFunctionGlobal, file: &FileReader) -> VortexResult<bool> {
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
    let row_range = 0..file.row_count;
    let mask = Mask::new_true(usize::try_from(file.row_count).unwrap_or(usize::MAX));
    let evaluation = file.reader.pruning_evaluation(&row_range, filter, mask)?;
    match evaluation.now_or_never() {
        Some(Ok(result_mask)) => Ok(result_mask.all_false()),
        _ => Ok(false),
    }
}

const FILE_CHANNEL_CAPACITY: usize = 16;

pub fn file_start_scan(
    bind: &TableFunctionBind,
    global: &TableFunctionGlobal,
    file: &FileReader,
    column_ids: &[u64],
    filters: cpp::duckdb_vx_table_filter_set,
) -> VortexResult<FileScan> {
    let file_row_number_column_pos = column_ids
        .iter()
        .position(|&id| id == FILE_ROW_NUMBER_COLUMN_IDX);
    let file_index_column_pos = column_ids
        .iter()
        .position(|&id| id == FILE_INDEX_COLUMN_IDX);

    let Filter {
        filter,
        row_selection,
        row_range,
        has_non_optional_filter,
        ..
    } = convert_filter(bind, column_ids, filters)?;
    if has_non_optional_filter {
        bind.has_non_optional_filter.store(true, Ordering::Relaxed);
    }

    let filter = filter
        .map(|expr| optimize_and_bind(expr, &bind.dtype))
        .transpose()?;

    let mut builder = ScanBuilder::new(SESSION.clone(), Arc::clone(&file.reader))
        .with_projection(global.bound_projection.clone())
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
    let total_splits = handles.len() as u64;

    let cache = Arc::new(ConversionCache::default());

    let pending = (!global.aggregates.is_empty()).then(|| Arc::clone(&global.pending));
    let (sender, receiver) = kanal::bounded_async(FILE_CHANNEL_CAPACITY);
    let driver = RUNTIME.handle().spawn(async move {
        for handle in handles {
            match handle.await {
                Ok(Some(array)) => {
                    if let Some(pending) = &pending {
                        pending.fetch_add(1, Ordering::Relaxed);
                    }
                    if sender.send(Ok((array, Arc::clone(&cache)))).await.is_err() {
                        // The receiver is gone: the scan was cancelled or the query ended.
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

    Ok(FileScan {
        receiver,
        total_splits,
        delivered: AtomicU64::new(0),
        file_row_number_column_pos,
        file_index_column_pos,
        aggregate_positions: global.aggregate_positions.clone(),
        _driver: driver,
    })
}

pub fn file_has_work(local: &TableFunctionLocal) -> bool {
    !local.exhausted
}

fn file_scan_aggregate(
    scan: &FileScan,
    global: &TableFunctionGlobal,
    local: &mut TableFunctionLocal,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let has_count_star = local.partials.len() < global.aggregates.len();
    let mut accumulated = 0u64;
    let mut rows = 0u64;
    loop {
        let Ok(item) = RUNTIME.block_on(scan.receiver.recv()) else {
            local.exhausted = true;
            break;
        };
        let (array, _cache) = item?;
        scan.delivered.fetch_add(1, Ordering::Relaxed);
        let array = convert_result(array, &mut ctx)?;

        for (position, partial) in scan
            .aggregate_positions
            .iter()
            .zip(local.partials.iter_mut())
        {
            partial.accumulate(array.unmasked_field(*position), &mut ctx)?;
        }
        rows += array.len() as u64;
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

pub fn file_scan(
    scan: &FileScan,
    global: &TableFunctionGlobal,
    local: &mut TableFunctionLocal,
    output: &mut DataChunkRef,
) -> VortexResult<()> {
    if !local.partials.is_empty() {
        return file_scan_aggregate(scan, global, local);
    }
    loop {
        if local.exporter.is_none() {
            let Ok(item) = RUNTIME.block_on(scan.receiver.recv()) else {
                local.exhausted = true;
                return Ok(());
            };
            let (array, cache) = item?;
            scan.delivered.fetch_add(1, Ordering::Relaxed);

            let mut ctx = SESSION.create_execution_ctx();
            let array = convert_result(array, &mut ctx)?;
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
    Ok(())
}

pub fn file_progress(reader: &FileScan) -> f64 {
    if scan.total_splits == 0 {
        return 100.0;
    }
    let delivered = scan.delivered.load(Ordering::Relaxed) as f64;
    100.0 * delivered / scan.total_splits as f64
}
