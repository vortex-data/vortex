// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::fmt::Formatter;
use std::fmt::{self};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use custom_labels::CURRENT_LABELSET;
use futures::StreamExt;
use itertools::Itertools;
use num_traits::AsPrimitive;
use static_assertions::assert_impl_all;
use tracing::debug;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::ScalarFn;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::stats::Precision;
use vortex::file::v2::FileStatsLayoutReader;
use vortex::io::kanal_ext::KanalExt as _;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::io::runtime::current::ThreadSafeIterator;
use vortex::layout::scan::multi::MultiLayoutChild;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex::metrics::tracing::get_global_labels;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::pack::Pack;
use vortex::scan::DataSource;
use vortex::scan::ScanRequest;
use vortex_utils::parallelism::get_available_parallelism;

use crate::RUNTIME;
use crate::SESSION;
use crate::column_statistics::ColumnStatistics;
use crate::column_statistics::ColumnStatisticsAggregate;
use crate::convert::try_from_bound_expression;
use crate::convert::try_from_projection_expression;
use crate::duckdb::BindInputRef;
use crate::duckdb::BindResultRef;
use crate::duckdb::DataChunkRef;
use crate::duckdb::DuckdbStringMapRef;
use crate::duckdb::ExpressionRef;
use crate::duckdb::TableInitInput;
use crate::duckdb::Value;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::multi_file::bind_multi_file_scan;
use crate::projection::DuckdbField;
use crate::projection::Filter;
use crate::projection::Projection;
use crate::projection::extract_schema_from_dtype;

pub struct TableFunctionBind {
    data_source: Arc<MultiLayoutDataSource>,
    filter_exprs: Vec<Expression>,
    column_fields: Vec<DuckdbField>,
    // There exists at least one non-optional table filter or at least one
    // complex filter is pushed down.
    has_non_optional_filter: AtomicBool,
}
assert_impl_all!(TableFunctionBind: Send, Clone);

impl Clone for TableFunctionBind {
    fn clone(&self) -> Self {
        Self {
            data_source: Arc::clone(&self.data_source),
            // filter_exprs are consumed once in `init_global`.
            filter_exprs: vec![],
            column_fields: self.column_fields.clone(),
            has_non_optional_filter: AtomicBool::new(
                self.has_non_optional_filter.load(Ordering::Relaxed),
            ),
        }
    }
}

impl fmt::Debug for TableFunctionBind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataSourceBindData")
            .field("column_fields", &self.column_fields)
            .field(
                "filter_exprs",
                &self
                    .filter_exprs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<String>>(),
            )
            .finish()
    }
}

impl<'a> TableInitInput<'a> {
    pub fn bind_data(&self) -> &TableFunctionBind {
        unsafe { &*self.input.bind_data.cast::<TableFunctionBind>() }
    }
}

type DataSourceIterator = ThreadSafeIterator<VortexResult<(ArrayRef, Arc<ConversionCache>)>>;

pub struct TableFunctionGlobal {
    iterator: DataSourceIterator,
    batch_id: AtomicU64,
    bytes_total: Arc<AtomicU64>,
    bytes_read: AtomicU64,
    file_index_column_pos: Option<usize>,
    file_row_number_column_pos: Option<usize>,
}
assert_impl_all!(TableFunctionGlobal: Send, Sync);

/// Per-thread scan state
pub struct TableFunctionLocal {
    iterator: DataSourceIterator,
    exporter: Option<ArrayExporter>,
    partition_index: u64,
    file_index: usize,
}

pub struct PartitionData {
    pub partition_index: u64,
    pub file_index_column_pos: Option<usize>,
    pub file_index: usize,
}

#[derive(Debug)]
pub enum Cardinality {
    /// Unknown number of rows
    Unknown,
    /// The exact number of rows.
    Exact(u64),
    /// An estimate of the number of rows.
    Estimate(u64),
}

pub fn bind(input: &BindInputRef, result: &mut BindResultRef) -> VortexResult<TableFunctionBind> {
    let data_source = bind_multi_file_scan(input)?;
    let column_fields = extract_schema_from_dtype(data_source.dtype())?;
    for fields in &column_fields {
        result.add_result_column(&fields.name, &fields.logical_type);
    }
    Ok(TableFunctionBind {
        data_source: Arc::new(data_source),
        filter_exprs: vec![],
        column_fields,
        has_non_optional_filter: AtomicBool::new(false),
    })
}

pub fn init_global(init_input: &TableInitInput) -> VortexResult<TableFunctionGlobal> {
    debug!(input=?init_input, "table function global input");

    let bind_data = init_input.bind_data();
    let column_ids = init_input.column_ids();
    let projection_ids = init_input.projection_ids();

    let Projection {
        projection,
        file_index_column_pos,
        file_row_number_column_pos,
    } = Projection::new(projection_ids, column_ids, &bind_data.column_fields);

    let Filter {
        filter,
        row_selection,
        row_range,
        file_selection,
        file_range,
        has_non_optional_filter,
    } = Filter::new(
        init_input.table_filter_set(),
        column_ids,
        &bind_data.column_fields,
        &bind_data.filter_exprs,
        bind_data.data_source.dtype(),
    )?;

    if has_non_optional_filter {
        init_input
            .bind_data()
            .has_non_optional_filter
            .store(true, Ordering::Relaxed);
    }

    debug!(
        %projection,
        filter = filter
            .as_ref()
            .map_or_else(|| "true".to_string(), |f| f.to_string()),
        ?row_selection,
        ?row_range,
        ?file_selection,
        ?file_range,
        "table function scan input"
    );

    let request = ScanRequest {
        projection,
        filter,
        ordered: file_row_number_column_pos.is_some(),
        selection: row_selection,
        row_range,
        partition_selection: file_selection,
        partition_range: file_range,
        limit: None,
    };

    let scan = RUNTIME.block_on(bind_data.data_source.scan(request))?;

    let num_workers = get_available_parallelism().unwrap_or(1);

    // We create an async bounded channel so that all thread-local workers can pull the next
    // available array chunk regardless of which partition it came from.
    let (tx, rx) = kanal::bounded_async(num_workers * 2);

    // We drive one partition per worker thread. Each partition is driven as a spawned task
    // that pushes array chunks into the shared channel as they are produced. This spawning
    // allows all worker threads to drive the polling of all partitions, and then return the
    // first available array chunk.
    let stream = scan
        .partitions()
        .map(move |partition| {
            let tx = tx.clone();
            RUNTIME.handle().spawn(async move {
                let partition = match partition {
                    Ok(partition) => partition,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };

                let cache = Arc::new(ConversionCache {
                    file_index: partition.index(),
                    ..Default::default()
                });

                let mut stream = match partition.execute() {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };
                while let Some(item) = stream.next().await {
                    if tx
                        .send(item.map(|a| (a, Arc::clone(&cache))))
                        .await
                        .is_err()
                    {
                        // Exit early if the receiver has been dropped, which happens when the
                        // scan is complete or if an error has occurred in another partition.
                        return;
                    }
                }
            })
        })
        .buffer_unordered(num_workers);

    // Spawn a task to drive the partition stream and push array chunks into the channel.
    RUNTIME.handle().spawn(stream.collect::<()>()).detach();

    let iterator = RUNTIME.block_on_stream_thread_safe(|_handle| rx.into_stream());

    Ok(TableFunctionGlobal {
        iterator,
        batch_id: AtomicU64::new(0),
        bytes_total: Arc::new(AtomicU64::new(0)),
        bytes_read: AtomicU64::new(0),
        file_index_column_pos,
        file_row_number_column_pos,
    })
}

pub fn init_local(global: &TableFunctionGlobal) -> TableFunctionLocal {
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

    TableFunctionLocal {
        iterator: global.iterator.clone(),
        exporter: None,
        partition_index: 0,
        file_index: 0,
    }
}

pub fn scan(
    local_state: &mut TableFunctionLocal,
    global_state: &TableFunctionGlobal,
    chunk: &mut DataChunkRef,
) -> VortexResult<()> {
    loop {
        if local_state.exporter.is_none() {
            let mut ctx = SESSION.create_execution_ctx();
            let Some(result) = local_state.iterator.next() else {
                return Ok(());
            };
            let (array_result, conversion_cache) = result?;
            let array_result = array_result.optimize_recursive(ctx.session())?;
            local_state.file_index = conversion_cache.file_index;

            let array_result: StructArray = if let Some(array) = array_result.as_opt::<Struct>() {
                array.into_owned()
            } else if let Some(array) = array_result.as_opt::<ScalarFn>()
                && let Some(pack_options) = array.scalar_fn().as_opt::<Pack>()
            {
                StructArray::new(
                    pack_options.names.clone(),
                    array.children(),
                    array.len(),
                    pack_options.nullability.into(),
                )
            } else {
                array_result.execute::<Canonical>(&mut ctx)?.into_struct()
            };

            local_state.exporter = Some(ArrayExporter::try_new(
                &array_result,
                &conversion_cache,
                ctx,
            )?);
            // Relaxed since there is no intra-instruction ordering required.
            local_state.partition_index = global_state.batch_id.fetch_add(1, Ordering::Relaxed);
        }

        let exporter = local_state
            .exporter
            .as_mut()
            .vortex_expect("error: exporter missing");
        let has_more_data = exporter.export(
            chunk,
            global_state.file_index_column_pos,
            global_state.file_row_number_column_pos,
        )?;

        global_state
            .bytes_read
            .fetch_add(chunk.len(), Ordering::Relaxed);

        if !has_more_data {
            // This exporter is fully consumed.
            local_state.exporter = None;
            local_state.partition_index = 0;
        } else {
            break;
        }
    }

    assert!(!chunk.is_empty());

    if let Some(pos) = global_state.file_index_column_pos {
        chunk
            .get_vector_mut(pos)
            .reference_value(&Value::from(local_state.file_index as u64));
    }

    Ok(())
}

/// Scan progress as a percentage (0.0–100.0).
pub fn table_scan_progress(global_state: &TableFunctionGlobal) -> f64 {
    progress(&global_state.bytes_read, &global_state.bytes_total)
}

pub fn pushdown_complex_filter(
    bind_data: &mut TableFunctionBind,
    expr: &ExpressionRef,
) -> VortexResult<bool> {
    debug!(%expr, "pushing down expression");

    let Some(expr) = try_from_bound_expression(expr)? else {
        debug!(%expr, "failed to push down expression");
        return Ok(false);
    };

    // Duckdb calls pushdown_complex_filter during planning phase.
    // If all filters are pushed down, duckdb enables a LEFT_DELIM_JOIN ->
    // COMPARISON_JOIN (HASH_JOIN) optimization:
    // duckdb/src/optimizer/deliminator.cpp: Deliminator::HasSelection,
    // Deliminator::Optimize.
    //
    // This leads to a massive regression on tpch sf=10 q17 and other
    // benchmarks.
    //
    // This bug is reported to Duckdb
    // https://github.com/duckdb/duckdb/issues/22669
    //
    // As a hack, report equality filters as not pushed.
    // We can also report only the first filter as not pushed, but this
    // has a negative performance impact.
    let report_pushed = !expr
        .as_opt::<Binary>()
        .map(|op| *op == Operator::Eq)
        .unwrap_or(false);

    // Only table filters may be optional, any complex filter is
    // non-optional by definition.
    bind_data
        .has_non_optional_filter
        .store(true, Ordering::Relaxed);

    debug!(%expr, report_pushed, "pushed down expression");
    bind_data.filter_exprs.push(expr);
    Ok(report_pushed)
}

pub fn pushdown_projection_expression(
    bind_data: &mut TableFunctionBind,
    expr: &ExpressionRef,
    projection_id: usize,
) -> VortexResult<bool> {
    let field = &bind_data.column_fields[projection_id];
    debug!(%expr, %projection_id, col_name=field.name, "pushing down projection expression");
    match try_from_projection_expression(expr, field)? {
        None => {
            debug!(%expr, "failed to push down expression");
            Ok(false)
        }
        Some(vx_expr) => {
            debug!(%expr, "pushed down expression");
            bind_data.column_fields[projection_id].projection_expr = Some(vx_expr);
            Ok(true)
        }
    }
}

/// Get column-wise statistics. Available only if we're reading a single file.
pub fn statistics(bind_data: &TableFunctionBind, column_index: usize) -> Option<ColumnStatistics> {
    let children = bind_data.data_source.children();
    // Otherwise we'd have to open all files eagerly which is a performance
    // regression. Duckdb's Parquet reader only gets metadata for multiple
    // files with a UNION BY NAME and we don't support it (yet)
    // See duckdb/common/multi_file/multi_file_function.hpp#L691
    if children.len() != 1 {
        return None;
    }
    let MultiLayoutChild::Opened { reader, .. } = &children[0] else {
        return None;
    };
    let stats_sets = match reader.as_any().downcast_ref::<FileStatsLayoutReader>() {
        Some(inner) => inner.file_stats().stats_sets(),
        None => return None,
    };
    let stats_aggregate = ColumnStatisticsAggregate::new(&stats_sets[column_index]);
    let dtype = bind_data.column_fields[column_index].dtype.clone();
    Some(ColumnStatistics::from(&stats_aggregate, dtype))
}

/// Duckdb requires post-filter cardinality estimates, otherwise join planner
/// may flip join sides which is a huge regression for some queries i.e. 1000x
/// for tpcds 85.
///
/// See duckdb/src/optimizer/join_order/relation_statistics_helper.cpp
///
/// As we don't report distinct values (same as Parquet), the only heuristic
/// duckdb uses is a 0.2 filter if there is any non-optional filter. We mimic it
/// here.
const DEFAULT_SELECTIVITY: f64 = 0.2;
pub fn cardinality(bind_data: &TableFunctionBind) -> Cardinality {
    let has_non_optional_filter = bind_data.has_non_optional_filter.load(Ordering::Relaxed);
    match bind_data.data_source.row_count() {
        Precision::Exact(v) => {
            if !has_non_optional_filter {
                return Cardinality::Exact(v);
            }
            let post_cardinality = v as f64 * DEFAULT_SELECTIVITY;
            let post_cardinality: u64 = post_cardinality.as_();
            Cardinality::Estimate(max(1, post_cardinality))
        }
        Precision::Inexact(v) => {
            if !has_non_optional_filter {
                return Cardinality::Estimate(v);
            }
            let post_cardinality = v as f64 * DEFAULT_SELECTIVITY;
            let post_cardinality: u64 = post_cardinality.as_();
            Cardinality::Estimate(max(1, post_cardinality))
        }
        Precision::Absent => Cardinality::Unknown,
    }
}

pub fn get_partition_data(
    global_init_data: &TableFunctionGlobal,
    local_init_data: &mut TableFunctionLocal,
) -> PartitionData {
    PartitionData {
        partition_index: local_init_data.partition_index,
        file_index_column_pos: global_init_data.file_index_column_pos,
        file_index: local_init_data.file_index,
    }
}

pub fn to_string(bind_data: &TableFunctionBind, map: &mut DuckdbStringMapRef) {
    map.push("Function", "Vortex Scan");
    if !bind_data.filter_exprs.is_empty() {
        let mut filters = bind_data.filter_exprs.iter().map(|f| format!("{f}"));
        map.push("Filters", &filters.join("\n"));
    }
    let projections = bind_data
        .column_fields
        .iter()
        .filter_map(|field| {
            field
                .projection_expr
                .as_ref()
                .map(|expr| format!("{}: {expr}", field.name))
        })
        .join("\n");
    if !projections.is_empty() {
        map.push("SELECT projections", &projections);
    }
}

fn progress(bytes_read: &AtomicU64, bytes_total: &AtomicU64) -> f64 {
    let read = bytes_read.load(Ordering::Relaxed);
    let mut total = bytes_total.load(Ordering::Relaxed);
    total += (total == 0) as u64;
    read as f64 / total as f64 * 100.
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;

    use crate::table_function::progress;

    #[test]
    fn test_table_scan_progress() {
        let bytes_total = AtomicU64::new(100);
        let bytes_read = AtomicU64::new(0);

        assert_eq!(progress(&bytes_read, &bytes_total), 0.0);

        bytes_read.fetch_add(100, Relaxed);
        assert_eq!(progress(&bytes_read, &bytes_total), 100.);

        bytes_total.fetch_add(100, Relaxed);
        assert!((progress(&bytes_read, &bytes_total) - 50.).abs() < f64::EPSILON);
    }
}
