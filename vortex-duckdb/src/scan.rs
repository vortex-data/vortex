// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::ffi::CString;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use custom_labels::CURRENT_LABELSET;
use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use futures::stream::SelectAll;
use itertools::Itertools;
use num_traits::AsPrimitive;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::ScalarFnVTable;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::StructVTable;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::dtype::FieldNames;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression as VortexExpression;
use vortex::expr::Pack;
use vortex::expr::and_collect;
use vortex::expr::col;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::filesystem::FileListing;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::ThreadSafeIterator;
use vortex::metrics::tracing::get_global_labels;
use vortex::session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::RUNTIME;
use crate::SESSION;
use crate::convert::try_from_bound_expression;
use crate::convert::try_from_table_filter;
use crate::duckdb::BindInput;
use crate::duckdb::BindResult;
use crate::duckdb::Cardinality;
use crate::duckdb::ClientContext;
use crate::duckdb::DataChunk;
use crate::duckdb::Expression;
use crate::duckdb::ExtractedValue;
use crate::duckdb::LogicalType;
use crate::duckdb::TableFunction;
use crate::duckdb::TableInitInput;
use crate::duckdb::VirtualColumnsResult;
use crate::duckdb::footer_cache::FooterCache;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::filesystem::resolve_filesystem;

pub struct VortexBindData {
    file_system: FileSystemRef,
    first_file: VortexFile,
    filter_exprs: Vec<VortexExpression>,
    files: Vec<FileListing>,
    column_names: Vec<String>,
    column_types: Vec<LogicalType>,
}

impl Clone for VortexBindData {
    /// `VortexBindData` is cloned in case of multiple scan nodes.
    fn clone(&self) -> Self {
        Self {
            file_system: self.file_system.clone(),
            first_file: self.first_file.clone(),
            // filter_expr don't need to be cloned as they are consumed once in `init_global`.
            filter_exprs: vec![],
            files: self.files.clone(),
            column_names: self.column_names.clone(),
            column_types: self.column_types.clone(),
        }
    }
}

impl Debug for VortexBindData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexBindData")
            .field("file_system", &self.file_system)
            .field("file_urls", &self.files)
            .field("column_names", &self.column_names)
            .field("column_types", &self.column_types)
            .field("filter_expr", &self.filter_exprs)
            .finish()
    }
}

pub struct VortexGlobalData {
    iterator: ThreadSafeIterator<VortexResult<(ArrayRef, Arc<ConversionCache>)>>,
    batch_id: AtomicU64,
    ctx: ExecutionCtx,
}

pub struct VortexLocalData {
    iterator: ThreadSafeIterator<VortexResult<(ArrayRef, Arc<ConversionCache>)>>,
    exporter: Option<ArrayExporter>,
    // The unique batch id the of the last chunk exported via scan()
    batch_id: Option<u64>,
}

#[derive(Debug)]
pub struct VortexTableFunction;

/// Extracts the schema from a Vortex file.
fn extract_schema_from_vortex_file(
    file: &VortexFile,
) -> VortexResult<(Vec<String>, Vec<LogicalType>)> {
    let dtype = file.dtype();

    // For now, we assume the top-level type to be a struct.
    let struct_dtype = dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("Vortex file must contain a struct array at the top level"))?;

    let mut column_names = Vec::new();
    let mut column_types = Vec::new();

    for (field_name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
        let logical_type = LogicalType::try_from(&field_dtype)?;
        column_names.push(field_name.to_string());
        column_types.push(logical_type);
    }

    Ok((column_names, column_types))
}

/// Creates a projection expression based on the table initialization input.
fn extract_projection_expr(init: &TableInitInput<VortexTableFunction>) -> VortexExpression {
    let projection_ids = init.projection_ids().unwrap_or(&[]);
    let column_ids = init.column_ids();

    select(
        projection_ids
            .iter()
            .map(|p| {
                let idx: usize = p.as_();
                let val: usize = column_ids[idx].as_();
                val
            })
            .map(|idx| {
                init.bind_data()
                    .column_names
                    .get(idx)
                    .vortex_expect("prune idx in column names")
            })
            .map(|s| Arc::from(s.as_str()))
            .collect::<FieldNames>(),
        root(),
    )
}

/// Creates a table filter expression from the table filter set.
fn extract_table_filter_expr(
    init: &TableInitInput<VortexTableFunction>,
    column_ids: &[u64],
) -> VortexResult<Option<VortexExpression>> {
    let mut table_filter_exprs: HashSet<VortexExpression> =
        if let Some(filter) = init.table_filter_set() {
            filter
                .into_iter()
                .map(|(idx, ex)| {
                    let idx_u: usize = idx.as_();
                    let col_idx: usize = column_ids[idx_u].as_();
                    let name = init
                        .bind_data()
                        .column_names
                        .get(col_idx)
                        .vortex_expect("exists");
                    try_from_table_filter(
                        &ex,
                        &col(name.as_str()),
                        init.bind_data().first_file.dtype(),
                    )
                })
                .collect::<VortexResult<Option<HashSet<_>>>>()?
                .unwrap_or_else(HashSet::new)
        } else {
            HashSet::new()
        };

    table_filter_exprs.extend(init.bind_data().filter_exprs.clone());
    Ok(and_collect(table_filter_exprs.into_iter().collect_vec()))
}

// taken from duckdb/common/constants.h COLUMN_IDENTIFIER_EMPTY
// This is used by duckdb whenever there is no projection id in a logical_get node.
// For some reason we cannot return an empty DataChunk and duckdb will look for the virtual column
// with this index and create a data chunk with a single vector of that type.
static EMPTY_COLUMN_IDX: u64 = 18446744073709551614;
static EMPTY_COLUMN_NAME: &str = "";

impl TableFunction for VortexTableFunction {
    type BindData = VortexBindData;
    type GlobalState = VortexGlobalData;
    type LocalState = VortexLocalData;

    const PROJECTION_PUSHDOWN: bool = true;
    const FILTER_PUSHDOWN: bool = true;
    const FILTER_PRUNE: bool = true;

    /// Input parameter types of the `vortex_scan` table function.
    ///
    // `vortex_scan` takes a single file glob parameter.
    fn parameters() -> Vec<LogicalType> {
        vec![LogicalType::varchar()]
    }

    fn bind(
        ctx: &ClientContext,
        input: &BindInput,
        result: &mut BindResult,
    ) -> VortexResult<Self::BindData> {
        let glob_url_parameter = input
            .get_parameter(0)
            .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

        // Parse the URL and separate the base URL (keep scheme, host, etc.) from the path.
        let glob_url_str = glob_url_parameter.as_ref().as_string();
        let glob_url = match Url::parse(glob_url_str.as_str()) {
            Ok(url) => url,
            Err(_) => {
                // Otherwise, we assume it's a file path.
                let path = if !glob_url_str.as_str().starts_with("/") {
                    // We cannot use Path::canonicalize to resolve relative paths since it
                    // requires the file to exist, and the glob may contain wildcards. Instead,
                    // we resolve relative paths against the current working directory.
                    let current_dir = std::env::current_dir().map_err(|e| {
                        vortex_err!(
                            "Cannot get current working directory to resolve relative path {}: {}",
                            glob_url_str.as_str(),
                            e
                        )
                    })?;
                    current_dir.join(glob_url_str.as_str())
                } else {
                    Path::new(glob_url_str.as_str()).to_path_buf()
                };

                Url::from_file_path(path).map_err(|_| {
                    vortex_err!("Cannot convert path to URL: {}", glob_url_str.as_str())
                })?
            }
        };

        let mut base_url = glob_url.clone();
        base_url.set_path("");

        let fs: FileSystemRef = resolve_filesystem(&base_url, ctx)?;

        // Read the vortex_max_threads setting from DuckDB configuration
        let max_threads_cstr = CString::new("vortex_max_threads")
            .map_err(|e| vortex_err!("Invalid setting name: {}", e))?;
        let max_threads = ctx
            .try_get_current_setting(&max_threads_cstr)
            .and_then(|v| match v.as_ref().extract() {
                ExtractedValue::UBigInt(val) => usize::try_from(val).ok(),
                ExtractedValue::BigInt(val) if val > 0 => usize::try_from(val as u64).ok(),
                _ => None,
            })
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });

        tracing::trace!("running scan with max_threads {max_threads}");

        let glob_pattern = glob_url
            .path()
            .strip_prefix("/")
            .unwrap_or_else(|| glob_url.path());
        let files: Vec<FileListing> = RUNTIME
            .block_on_stream(fs.glob(glob_pattern)?)
            .try_collect()?;

        // The first file is skipped in `create_file_paths_queue`.
        let Some(first_file_listing) = files.first() else {
            vortex_bail!("No files matched the glob");
        };

        let footer_cache = FooterCache::new(ctx.object_cache());
        let entry = footer_cache.entry(&first_file_listing.path);
        let fs2 = fs.clone();
        let first_file = RUNTIME.block_on(async move {
            let options = entry
                .apply_to_file(SESSION.open_options())
                .with_some_file_size(first_file_listing.size);
            let read_at = fs2.open_read(&first_file_listing.path).await?;
            let file = options.open(read_at).await?;
            entry.put_if_absent(|| file.footer().clone());
            VortexResult::Ok(file)
        })?;

        let (column_names, column_types) = extract_schema_from_vortex_file(&first_file)?;

        // Add result columns based on the extracted schema.
        for (column_name, column_type) in column_names.iter().zip(&column_types) {
            result.add_result_column(column_name, column_type);
        }

        Ok(VortexBindData {
            file_system: fs,
            files,
            first_file,
            filter_exprs: vec![],
            column_names,
            column_types,
        })
    }

    fn scan(
        _client_context: &ClientContext,
        _bind_data: &Self::BindData,
        local_state: &mut Self::LocalState,
        global_state: &mut Self::GlobalState,
        chunk: &mut DataChunk,
    ) -> VortexResult<()> {
        loop {
            if local_state.exporter.is_none() {
                let Some(result) = local_state.iterator.next() else {
                    return Ok(());
                };

                let (array_result, conversion_cache) = result?;

                let array_result = array_result.optimize_recursive()?;
                let array_result = if let Some(array) = array_result.as_opt::<StructVTable>() {
                    array.clone()
                } else if let Some(array) = array_result.as_opt::<ScalarFnVTable>()
                    && let Some(pack_options) = array.scalar_fn().as_opt::<Pack>()
                {
                    StructArray::new(
                        pack_options.names.clone(),
                        array.children(),
                        array.len(),
                        pack_options.nullability.into(),
                    )
                } else {
                    array_result
                        .execute::<Canonical>(&mut global_state.ctx)?
                        .into_struct()
                };

                local_state.exporter = Some(ArrayExporter::try_new(
                    &array_result,
                    &conversion_cache,
                    &mut global_state.ctx,
                )?);
                // Relaxed since there is no intra-instruction ordering required.
                local_state.batch_id = Some(global_state.batch_id.fetch_add(1, Ordering::Relaxed));
            }

            let exporter = local_state
                .exporter
                .as_mut()
                .vortex_expect("error: exporter missing");

            let has_more_data = exporter.export(chunk)?;

            if !has_more_data {
                // This exporter is fully consumed.
                local_state.exporter = None;
                local_state.batch_id = None;
            } else {
                break;
            }
        }

        assert!(!chunk.is_empty());

        Ok(())
    }

    fn init_global(init_input: &TableInitInput<Self>) -> VortexResult<Self::GlobalState> {
        let bind_data = init_input.bind_data();
        let projection_expr = extract_projection_expr(init_input);
        let filter_expr = extract_table_filter_expr(init_input, init_input.column_ids())?;

        tracing::trace!(
            "Global init Vortex scan SELECT {} WHERE {}",
            &projection_expr,
            filter_expr
                .as_ref()
                .map_or_else(|| "true".to_string(), |f| f.to_string())
        );

        let client_context = init_input.client_context()?;
        let object_cache = client_context.object_cache();

        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let handle = RUNTIME.handle();
        let fs = bind_data.file_system.clone();
        let first_file = bind_data.first_file.clone();
        let scan_streams = stream::iter(bind_data.files.clone())
            .enumerate()
            .map(move |(idx, file_listing)| {
                let fs = fs.clone();
                let first_file = first_file.clone();
                let filter_expr = filter_expr.clone();
                let projection_expr = projection_expr.clone();
                let conversion_cache = Arc::new(ConversionCache::new(idx as u64));
                let object_cache = object_cache;

                handle
                    .spawn(async move {
                        let vxf = if idx == 0 {
                            // The first path from `file_paths` is skipped as
                            // the first file was already opened during bind.
                            Ok(first_file)
                        } else {
                            let cache = FooterCache::new(object_cache);
                            let entry = cache.entry(&file_listing.path);
                            let file = entry
                                .apply_to_file(SESSION.open_options())
                                .with_some_file_size(file_listing.size)
                                .open(fs.open_read(&file_listing.path).await?)
                                .await?;
                            entry.put_if_absent(|| file.footer().clone());
                            VortexResult::Ok(file)
                        }?;

                        if let Some(ref filter) = filter_expr
                            && vxf.can_prune(filter)?
                        {
                            return Ok(None);
                        };

                        let scan = vxf
                            .scan()?
                            .with_some_filter(filter_expr)
                            .with_projection(projection_expr)
                            .with_ordered(false)
                            .map(move |split| Ok((split, conversion_cache.clone())))
                            .into_stream()?
                            .boxed();

                        Ok(Some(scan))
                    })
                    .boxed()
            })
            // Open up to num_workers * 2 files concurrently so we always have one ready to go.
            .buffer_unordered(num_workers * 2)
            .filter_map(|result| async move { result.transpose() });

        Ok(VortexGlobalData {
            iterator: RUNTIME.block_on_stream_thread_safe(move |_| MultiScan {
                streams: scan_streams.boxed(),
                streams_finished: false,
                select_all: Default::default(),
                max_concurrency: num_workers * 2,
            }),
            batch_id: AtomicU64::new(0),
            // TODO(joe): fetch this from somewhere??.
            ctx: ExecutionCtx::new(VortexSession::default()),
        })
    }

    fn init_local(
        _init: &TableInitInput<Self>,
        global: &mut Self::GlobalState,
    ) -> VortexResult<Self::LocalState> {
        unsafe {
            use custom_labels::sys;

            if sys::current().is_null() {
                let ls = sys::new(0);
                sys::replace(ls);
            };
        }

        let global_labels = get_global_labels();

        for (key, value) in global_labels {
            CURRENT_LABELSET.set(key, value);
        }

        Ok(VortexLocalData {
            iterator: global.iterator.clone(),
            exporter: None,
            batch_id: None,
        })
    }

    fn pushdown_complex_filter(
        bind_data: &mut Self::BindData,
        expr: &Expression,
    ) -> VortexResult<bool> {
        let Some(expr) = try_from_bound_expression(expr)? else {
            return Ok(false);
        };
        bind_data.filter_exprs.push(expr);
        // It seems like there is a regression in the DuckDB planner we actually delete filters??
        // TODO(joe): file and issue and fix.
        Ok(false)
    }

    fn cardinality(bind_data: &Self::BindData) -> Cardinality {
        if bind_data.files.len() == 1 {
            Cardinality::Maximum(bind_data.first_file.row_count())
        } else {
            // This is the same behavior as DuckDB's Parquet extension, although we could
            // test multiplying the row count by the number of files.
            Cardinality::Estimate(
                max(bind_data.first_file.row_count(), 1) * bind_data.files.len() as u64,
            )
        }
    }

    fn partition_data(
        _bind_data: &Self::BindData,
        _global_init_data: &mut Self::GlobalState,
        _local_init_data: &mut Self::LocalState,
    ) -> VortexResult<u64> {
        _local_init_data
            .batch_id
            .ok_or_else(|| vortex_err!("batch id missing, no batches exported"))
    }

    fn to_string(bind_data: &Self::BindData) -> Option<Vec<(String, String)>> {
        let mut result = Vec::new();

        // Add function name
        result.push(("Function".to_string(), "Vortex Scan".to_string()));

        // Add file information
        if !bind_data.files.is_empty() {
            result.push(("Files".to_string(), bind_data.files.len().to_string()));
        }

        // Add filter information
        if !bind_data.filter_exprs.is_empty() {
            let mut filters = bind_data.filter_exprs.iter().map(|f| format!("{}", f));
            result.push(("Filters".to_string(), filters.join(" /\\\n")));
        }
        // NOTE: Projection is already printed by the planner.

        Some(result)
    }

    fn virtual_columns(_bind_data: &Self::BindData, result: &mut VirtualColumnsResult) {
        result.register(EMPTY_COLUMN_IDX, EMPTY_COLUMN_NAME, &LogicalType::bool());
    }
}

struct MultiScan<'rt, T> {
    // A stream-of-streams of scan results.
    streams: BoxStream<'rt, VortexResult<BoxStream<'rt, VortexResult<T>>>>,
    streams_finished: bool,
    // The SelectAll used to drive the inner streams.
    select_all: SelectAll<BoxStream<'rt, VortexResult<T>>>,
    // The maximum number of streams to be driving concurrently.
    max_concurrency: usize,
}

impl<'rt, T: 'rt> Stream for MultiScan<'rt, T> {
    type Item = VortexResult<T>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;

        loop {
            // First, try to pull from the SelectAll of active streams.
            // This means we prefer to complete existing work before starting new work, unless it
            // all returns Poll::Pending.
            match this.select_all.poll_next_unpin(cx) {
                Poll::Ready(None) => {
                    if this.streams_finished {
                        // All streams are done
                        return Poll::Ready(None);
                    }
                }
                Poll::Ready(Some(result)) => return Poll::Ready(Some(result)),
                Poll::Pending => {
                    // None of the active streams are ready right now.
                }
            }

            // If all current streams returned `Poll::Pending`, then we try to fetch the next
            // stream to drive. The idea here is to ensure our executors are always busy with
            // CPU work by driving as many streams necessary to keep the I/O queues full.
            if this.select_all.len() < this.max_concurrency {
                match Pin::new(&mut this.streams).poll_next(cx) {
                    Poll::Ready(Some(Ok(stream))) => {
                        // Add the new stream to SelectAll, and continue the loop to poll it.
                        this.select_all.push(stream);
                        continue;
                    }
                    Poll::Ready(Some(Err(e))) => {
                        // Error opening one of the streams
                        return Poll::Ready(Some(Err(e)));
                    }
                    Poll::Ready(None) => {
                        // No more streams available from the source
                        this.streams_finished = true;
                        if this.select_all.is_empty() {
                            // No active streams, so we're done.
                            return Poll::Ready(None);
                        }
                        return Poll::Pending;
                    }
                    Poll::Pending => {
                        // Can't get more streams right now
                        return Poll::Pending;
                    }
                }
            } else {
                // We have enough active streams, so just wait for one of them to yield.
                return Poll::Pending;
            }
        }
    }
}
