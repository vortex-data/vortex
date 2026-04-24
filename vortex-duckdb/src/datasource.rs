// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reusable logic for driving a [`DataSourceRef`] scan through DuckDB's table function interface.
//!
//! Table functions that resolve to a [`DataSourceRef`] can implement [`DataSourceTableFunction`]
//! to get a blanket [`TableFunction`] implementation covering init, scan, progress, filter
//! pushdown, cardinality, and partitioning.

use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use custom_labels::CURRENT_LABELSET;
use futures::StreamExt;
use itertools::Itertools;
use num_traits::AsPrimitive;
use tracing::debug;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ScalarFn;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::col;
use vortex::expr::root;
use vortex::expr::select;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::file::v2::FileStatsLayoutReader;
use vortex::io::kanal_ext::KanalExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::ThreadSafeIterator;
use vortex::layout::scan::multi::MultiLayoutChild;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex::metrics::tracing::get_global_labels;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::pack::Pack;
use vortex::scan::DataSource;
use vortex::scan::ScanRequest;
use vortex_utils::aliases::hash_set::HashSet;
use vortex_utils::parallelism::get_available_parallelism;

use crate::RUNTIME;
use crate::SESSION;
use crate::convert::ToDuckDBScalar;
use crate::convert::try_from_bound_expression;
use crate::convert::try_from_table_filter;
use crate::duckdb::BindInputRef;
use crate::duckdb::BindResultRef;
use crate::duckdb::Cardinality;
use crate::duckdb::ClientContextRef;
use crate::duckdb::ColumnStatistics;
use crate::duckdb::DataChunkRef;
use crate::duckdb::DuckdbStringMapRef;
use crate::duckdb::ExpressionRef;
use crate::duckdb::LogicalType;
use crate::duckdb::TableFilterSetRef;
use crate::duckdb::TableFunction;
use crate::duckdb::TableInitInput;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;

/// Taken from
/// https://github.com/duckdb/duckdb/blob/dc11eadd8f0a7c600f0034810706605ebe10d5b9/src/include/duckdb/common/constants.hpp#L44
///
/// If DuckDB requests a zero-column projection from read_vortex like count(*),
/// its planner tries to get any column:
/// https://github.com/duckdb/duckdb/blob/dc11eadd8f0a7c600f0034810706605ebe10d5b9/src/planner/operator/logical_get.cpp#L149
///
/// If you define COLUMN_IDENTIFIER_EMPTY, planner takes it, otherwise the
/// first column. As we don't want to fill the output chunk and we can leave
/// it uninitialized in this case, we define COLUMN_IDENTIFIER_EMPTY as a
/// virtual column.
/// See virtual_columns in vortex-duckdb/cpp/table_function.cpp
static EMPTY_COLUMN_IDX: u64 = 18446744073709551614;

/// A trait for table functions that resolve to a [`DataSourceRef`].
///
/// Implementors only need to define how parameters are declared and how binding produces a
/// data source. All other [`TableFunction`] methods (init, scan, progress, filter pushdown,
/// cardinality, partitioning) are provided by a blanket implementation.
pub(crate) trait DataSourceTableFunction: Sized + Debug {
    /// Positional parameters
    fn parameters() -> Vec<LogicalType>;

    /// Bind the table function and return a [`DataSourceRef`].
    fn bind(ctx: &ClientContextRef, input: &BindInputRef) -> VortexResult<MultiLayoutDataSource>;
}

#[derive(Debug, Clone)]
struct DuckdbField {
    name: String,
    logical_type: LogicalType,
    dtype: DType,
}

/// Bind data produced by a [`DataSourceTableFunction`].
pub struct DataSourceBindData {
    data_source: Arc<MultiLayoutDataSource>,
    filter_exprs: Vec<Expression>,
    column_fields: Vec<DuckdbField>,
}

impl Clone for DataSourceBindData {
    fn clone(&self) -> Self {
        Self {
            data_source: Arc::clone(&self.data_source),
            // filter_exprs are consumed once in `init_global`.
            filter_exprs: vec![],
            column_fields: self.column_fields.clone(),
        }
    }
}

impl Debug for DataSourceBindData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

type DataSourceIterator = ThreadSafeIterator<VortexResult<(ArrayRef, Arc<ConversionCache>)>>;

/// Global scan state for driving a `DataSource` scan through DuckDB.
pub struct DataSourceGlobal {
    iterator: DataSourceIterator,
    batch_id: AtomicU64,
    bytes_total: Arc<AtomicU64>,
    bytes_read: AtomicU64,
}

/// Per-thread local scan state.
pub struct DataSourceLocal {
    iterator: DataSourceIterator,
    exporter: Option<ArrayExporter>,
    /// The unique batch id of the last chunk exported via scan().
    batch_id: Option<u64>,
}

/// Returns scan progress as a percentage (0.0–100.0).
fn progress(bytes_read: &AtomicU64, bytes_total: &AtomicU64) -> f64 {
    let read = bytes_read.load(Ordering::Relaxed);
    let mut total = bytes_total.load(Ordering::Relaxed);
    total += (total == 0) as u64;
    read as f64 / total as f64 * 100.
}

impl ColumnStatistics {
    fn from(stats: &ColumnStatisticsAggregate, dtype: DType) -> Self {
        let min = stats.min.as_ref().map(|value| {
            let value = value.clone();
            Scalar::try_new(dtype.clone(), Some(value))
                .vortex_expect("scalar dtype and value are incompatible")
                .try_to_duckdb_scalar()
                .vortex_expect("can't convert Scalar to duckdb Value")
        });
        let max = stats.max.as_ref().map(|value| {
            Scalar::try_new(dtype.clone(), Some(value.clone()))
                .vortex_expect("scalar dtype and value are incompatible")
                .try_to_duckdb_scalar()
                .vortex_expect("can't convert Scalar to duckdb Value")
        });

        let max_string_length = stats
            .max_string_length
            .map_or(0, |len| (1u64 << 63) | (len as u64));

        // Useful estimate if we didn't get null count stats
        let has_null = stats.has_null && dtype.is_nullable();

        Self {
            min,
            max,
            max_string_length,
            has_null,
        }
    }
}

#[derive(Default)]
pub struct ColumnStatisticsAggregate {
    pub min: Option<ScalarValue>,
    pub max: Option<ScalarValue>,
    pub max_string_length: Option<u32>,
    /// May be true if null count stat isn't present
    pub has_null: bool,
}

impl ColumnStatisticsAggregate {
    pub fn new(stats: &StatsSet) -> Self {
        let min = match stats.get(Stat::Min) {
            Some(Precision::Exact(min)) => Some(min),
            _ => None,
        };
        let max = match stats.get(Stat::Max) {
            Some(Precision::Exact(max)) => Some(max),
            _ => None,
        };

        let max_string_length =
            if let Some(Precision::Exact(value)) = stats.get(Stat::UncompressedSizeInBytes) {
                // DuckDB's string length is u32
                #[allow(clippy::cast_possible_truncation)]
                Some(value.as_primitive().as_u64().vortex_expect("not a u64") as u32)
            } else {
                None
            };

        let has_null = match stats.get(Stat::NullCount) {
            Some(Precision::Exact(cnt)) => {
                cnt.as_primitive().as_u64().vortex_expect("not a u64") > 0
            }
            _ => true,
        };

        Self {
            min,
            max,
            max_string_length,
            has_null,
        }
    }
}

// ---------------------------------------------------------------------------
// Blanket TableFunction implementation for any DataSourceTableFunction
// ---------------------------------------------------------------------------

impl<T: DataSourceTableFunction> TableFunction for T {
    type BindData = DataSourceBindData;
    type GlobalState = DataSourceGlobal;
    type LocalState = DataSourceLocal;

    fn parameters() -> Vec<LogicalType> {
        T::parameters()
    }

    fn bind(
        ctx: &ClientContextRef,
        input: &BindInputRef,
        result: &mut BindResultRef,
    ) -> VortexResult<Self::BindData> {
        let data_source = T::bind(ctx, input)?;
        let column_fields = extract_schema_from_dtype(data_source.dtype())?;
        for fields in &column_fields {
            result.add_result_column(&fields.name, &fields.logical_type);
        }
        Ok(DataSourceBindData {
            data_source: Arc::new(data_source),
            filter_exprs: vec![],
            column_fields,
        })
    }

    fn init_global(init_input: &TableInitInput<Self>) -> VortexResult<Self::GlobalState> {
        debug!("table init input: {init_input:?}");

        let bind_data = init_input.bind_data();
        let column_ids = init_input.column_ids();
        let projection_ids = init_input.projection_ids();

        let projection_expr =
            extract_projection_expr(projection_ids, column_ids, &bind_data.column_fields);
        let filter_expr = extract_table_filter_expr(
            init_input.table_filter_set(),
            column_ids,
            &bind_data.column_fields,
            &bind_data.filter_exprs,
            bind_data.data_source.dtype(),
        )?;

        let filter_expr_str = filter_expr
            .as_ref()
            .map_or_else(|| "true".to_string(), |f| f.to_string());
        debug!("Global init Vortex scan SELECT {projection_expr} WHERE {filter_expr_str}");

        let request = ScanRequest {
            projection: projection_expr,
            filter: filter_expr,
            ordered: false,
            ..Default::default()
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
                // We create a new conversion cache scoped to the partition, since there's no point
                // caching anything across partitions.
                let cache = Arc::new(ConversionCache::default());
                let tx = tx.clone();

                RUNTIME.handle().spawn(async move {
                    let mut stream = match partition.and_then(|p| p.execute()) {
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

        Ok(DataSourceGlobal {
            iterator,
            batch_id: AtomicU64::new(0),
            bytes_total: Arc::new(AtomicU64::new(0)),
            bytes_read: AtomicU64::new(0),
        })
    }

    fn init_local(
        _init: &TableInitInput<Self>,
        global: &Self::GlobalState,
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

        Ok(DataSourceLocal {
            iterator: global.iterator.clone(),
            exporter: None,
            batch_id: None,
        })
    }

    fn scan(
        _client_context: &ClientContextRef,
        _bind_data: &Self::BindData,
        local_state: &mut Self::LocalState,
        global_state: &Self::GlobalState,
        chunk: &mut DataChunkRef,
    ) -> VortexResult<()> {
        loop {
            if local_state.exporter.is_none() {
                let mut ctx = SESSION.create_execution_ctx();
                let Some(result) = local_state.iterator.next() else {
                    return Ok(());
                };
                let (array_result, conversion_cache) = result?;
                let array_result = array_result.optimize_recursive()?;

                let array_result: StructArray = if let Some(array) = array_result.as_opt::<Struct>()
                {
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
                local_state.batch_id = Some(global_state.batch_id.fetch_add(1, Ordering::Relaxed));
            }

            let exporter = local_state
                .exporter
                .as_mut()
                .vortex_expect("error: exporter missing");

            let has_more_data = exporter.export(chunk)?;
            global_state
                .bytes_read
                .fetch_add(chunk.len(), Ordering::Relaxed);

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

    fn table_scan_progress(
        _client_context: &ClientContextRef,
        _bind_data: &Self::BindData,
        global_state: &Self::GlobalState,
    ) -> f64 {
        progress(&global_state.bytes_read, &global_state.bytes_total)
    }

    fn pushdown_complex_filter(
        bind_data: &mut Self::BindData,
        expr: &ExpressionRef,
    ) -> VortexResult<bool> {
        tracing::debug!("Attempting to push down filter expression: {expr}");
        let Some(expr) = try_from_bound_expression(expr)? else {
            return Ok(false);
        };
        bind_data.filter_exprs.push(expr);

        // NOTE(ngates): Vortex does indeed run exact filters, so in theory we should return `true`
        //  here to tell DuckDB we've handled the filter. However, DuckDB applies some crude
        //  cardinality estimation heuristics (e.g. an equality filter => 20% selectivity) that
        //  means by returning false, DuckDB runs an additional filter (a little bit of overhead)
        //  but tends to end up with a better query plan.
        //  If we plumb row count estimation into the layout tree, perhaps we could use zone maps
        //  etc. to return estimates. But this function is probably called too late anyway. Maybe
        //  we need our own cardinality heuristics.
        Ok(false)
    }

    /// Get column-wise statistics. Available only if we're reading a single
    /// file.
    fn statistics(
        _client_context: &ClientContextRef,
        bind_data: &Self::BindData,
        column_index: usize,
    ) -> Option<ColumnStatistics> {
        let children = bind_data.data_source.children();
        // Otherwise we'd have to open all files eagerly which is a performance
        // regression. Duckdb's Parquet reader only gets metadata for multiple
        // files with a UNION BY NAME and we don't support it (yet)
        // See duckdb/common/multi_file/multi_file_function.hpp#L691
        if children.len() != 1 {
            return None;
        }
        let MultiLayoutChild::Opened(ref reader) = children[0] else {
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

    fn cardinality(bind_data: &Self::BindData) -> Cardinality {
        match bind_data.data_source.row_count() {
            Some(Precision::Exact(v)) => Cardinality::Maximum(v),
            Some(Precision::Inexact(v)) => Cardinality::Estimate(v),
            None => Cardinality::Unknown,
        }
    }

    fn partition_data(
        _bind_data: &Self::BindData,
        _global_init_data: &Self::GlobalState,
        local_init_data: &mut Self::LocalState,
    ) -> VortexResult<u64> {
        local_init_data
            .batch_id
            .ok_or_else(|| vortex_err!("batch id missing, no batches exported"))
    }

    fn to_string(bind_data: &Self::BindData, map: &mut DuckdbStringMapRef) {
        map.push("Function", "Vortex Scan");
        if !bind_data.filter_exprs.is_empty() {
            let mut filters = bind_data.filter_exprs.iter().map(|f| format!("{}", f));
            map.push("Filters", &filters.join(" /\\\n"));
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Extracts DuckDB column names and logical types from a Vortex struct DType.
fn extract_schema_from_dtype(dtype: &DType) -> VortexResult<Vec<DuckdbField>> {
    let struct_dtype = dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("Vortex file must contain a struct array at the top level"))?;

    let len = struct_dtype.names().len();
    let mut fields = Vec::with_capacity(len);

    for (field_name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
        let logical_type = LogicalType::try_from(&field_dtype)?;
        fields.push(DuckdbField {
            name: field_name.to_string(),
            logical_type,
            dtype: field_dtype,
        });
    }
    Ok(fields)
}

/// Creates a projection expression from raw projection/column ID slices and column names.
fn extract_projection_expr(
    projection_ids: Option<&[u64]>,
    column_ids: &[u64],
    column_fields: &[DuckdbField],
) -> Expression {
    // Projection ids may be empty, in which case you need to use projection_ids
    // See duckdb/src/planner/operator/logical_get.cpp#L168
    let (projection_ids, has_projection_ids) = match projection_ids {
        Some(ids) => (ids, true),
        None => (column_ids, false),
    };

    // duckdb index is u64 (size_t) but in Rust u64 and usize are different things.
    #[expect(clippy::cast_possible_truncation)]
    let names = projection_ids
        .iter()
        .filter(|p| **p != EMPTY_COLUMN_IDX)
        .map(|mut idx| {
            if has_projection_ids {
                idx = &column_ids[*idx as usize];
            }

            #[expect(clippy::cast_possible_truncation)]
            &column_fields
                .get(*idx as usize)
                .vortex_expect("prune idx in column names")
                .name
        })
        .map(|s| Arc::from(s.as_str()))
        .collect::<FieldNames>();

    select(names, root())
}

/// Creates a table filter expression from the table filter set, column metadata, additional
/// filter expressions, and the top-level DType.
fn extract_table_filter_expr(
    table_filter_set: Option<&TableFilterSetRef>,
    column_ids: &[u64],
    column_fields: &[DuckdbField],
    additional_filters: &[Expression],
    dtype: &DType,
) -> VortexResult<Option<Expression>> {
    let mut table_filter_exprs: HashSet<Expression> = if let Some(filter) = table_filter_set {
        filter
            .into_iter()
            .map(|(idx, ex)| {
                let idx_u: usize = idx.as_();
                let col_idx: usize = column_ids[idx_u].as_();
                let name = &column_fields.get(col_idx).vortex_expect("exists").name;
                try_from_table_filter(ex, &col(name.as_str()), dtype)
            })
            .collect::<VortexResult<Option<HashSet<_>>>>()?
            .unwrap_or_else(HashSet::new)
    } else {
        HashSet::new()
    };

    table_filter_exprs.extend(additional_filters.iter().cloned());
    Ok(and_collect(table_filter_exprs))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;

    use super::progress;

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
