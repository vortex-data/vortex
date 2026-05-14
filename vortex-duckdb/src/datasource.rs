// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reusable logic for driving a [`DataSourceRef`] scan through DuckDB's table function interface.
//!
//! Table functions that resolve to a [`DataSourceRef`] can implement [`DataSourceTableFunction`]
//! to get a blanket [`TableFunction`] implementation covering init, scan, progress, filter
//! pushdown, cardinality, and partitioning.

use std::fmt::Debug;
use std::ops::Range;
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
use vortex::expr::merge;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::select;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::file::v2::FileStatsLayoutReader;
use vortex::io::kanal_ext::KanalExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::ThreadSafeIterator;
use vortex::layout::layouts::row_idx::row_idx;
use vortex::layout::scan::multi::MultiLayoutChild;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex::metrics::tracing::get_global_labels;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::pack::Pack;
use vortex::scan::DataSource;
use vortex::scan::ScanRequest;
use vortex::scan::selection::Selection;
use vortex_utils::aliases::hash_set::HashSet;
use vortex_utils::parallelism::get_available_parallelism;

use crate::RUNTIME;
use crate::SESSION;
use crate::convert::ToDuckDBScalar;
use crate::convert::try_from_bound_expression;
use crate::convert::try_from_table_filter;
use crate::convert::try_from_virtual_column_filter;
use crate::duckdb::BindInputRef;
use crate::duckdb::BindResultRef;
use crate::duckdb::Cardinality;
use crate::duckdb::ClientContextRef;
use crate::duckdb::ColumnStatistics;
use crate::duckdb::DataChunkRef;
use crate::duckdb::DuckdbStringMapRef;
use crate::duckdb::ExpressionRef;
use crate::duckdb::LogicalType;
use crate::duckdb::PartitionData;
use crate::duckdb::TableFilterSetRef;
use crate::duckdb::TableFunction;
use crate::duckdb::TableInitInput;
use crate::duckdb::Value;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;

// See MultiFileReader for constants

/// "file_index" virtual column
static FILE_INDEX_COLUMN_IDX: u64 = 9223372036854775810;
/// "file_row_number" virtual column
static FILE_ROW_NUMBER_COLUMN_IDX: u64 = 9223372036854775809;

/// See duckdb/src/common/constants.cpp
fn is_virtual_column(id: u64) -> bool {
    id >= 9223372036854775808u64
}

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
    file_index_column_pos: Option<usize>,
    file_row_number_column_pos: Option<usize>,
}

/// Per-thread local scan state.
pub struct DataSourceLocal {
    iterator: DataSourceIterator,
    exporter: Option<ArrayExporter>,
    partition_index: u64,
    file_index: usize,
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
        debug!(input=?init_input, "table function global input");

        let bind_data = init_input.bind_data();
        let column_ids = init_input.column_ids();
        let projection_ids = init_input.projection_ids();

        let ProjectionWithVirtualColumns {
            projection,
            file_index_column_pos,
            file_row_number_column_pos,
        } = extract_projection_expr(projection_ids, column_ids, &bind_data.column_fields);

        let FilterWithVirtualColumns {
            filter,
            row_selection,
            row_range,
            file_selection,
            file_range,
        } = extract_table_filter_expr(
            init_input.table_filter_set(),
            column_ids,
            &bind_data.column_fields,
            &bind_data.filter_exprs,
            bind_data.data_source.dtype(),
        )?;

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

        Ok(DataSourceGlobal {
            iterator,
            batch_id: AtomicU64::new(0),
            bytes_total: Arc::new(AtomicU64::new(0)),
            bytes_read: AtomicU64::new(0),
            file_index_column_pos,
            file_row_number_column_pos,
        })
    }

    fn init_local(global: &Self::GlobalState) -> Self::LocalState {
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

        DataSourceLocal {
            iterator: global.iterator.clone(),
            exporter: None,
            partition_index: 0,
            file_index: 0,
        }
    }

    fn scan(
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
                let array_result = array_result.optimize_recursive(ctx.session())?;
                local_state.file_index = conversion_cache.file_index;

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

    fn table_scan_progress(global_state: &Self::GlobalState) -> f64 {
        progress(&global_state.bytes_read, &global_state.bytes_total)
    }

    fn pushdown_complex_filter(
        bind_data: &mut Self::BindData,
        expr: &ExpressionRef,
    ) -> VortexResult<bool> {
        debug!(%expr, "pushing down expression");
        let Some(expr) = try_from_bound_expression(expr)? else {
            debug!(%expr, "failed to push down expression");
            return Ok(false);
        };
        debug!(%expr, "pushed down expression");
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
    fn statistics(bind_data: &Self::BindData, column_index: usize) -> Option<ColumnStatistics> {
        let children = bind_data.data_source.children();
        // Otherwise we'd have to open all files eagerly which is a performance
        // regression. Duckdb's Parquet reader only gets metadata for multiple
        // files with a UNION BY NAME and we don't support it (yet)
        // See duckdb/common/multi_file/multi_file_function.hpp#L691
        if children.len() != 1 {
            return None;
        }
        let MultiLayoutChild::Opened(reader) = &children[0] else {
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
        global_init_data: &Self::GlobalState,
        local_init_data: &mut Self::LocalState,
    ) -> PartitionData {
        PartitionData {
            partition_index: local_init_data.partition_index,
            file_index_column_pos: global_init_data.file_index_column_pos,
            file_index: local_init_data.file_index,
        }
    }

    fn to_string(bind_data: &Self::BindData, map: &mut DuckdbStringMapRef) {
        map.push("Function", "Vortex Scan");
        if !bind_data.filter_exprs.is_empty() {
            let mut filters = bind_data.filter_exprs.iter().map(|f| format!("{}", f));
            map.push("Filters", &filters.join(" /\\\n"));
        }
    }
}

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

struct ProjectionWithVirtualColumns {
    projection: Expression,
    file_index_column_pos: Option<usize>,
    file_row_number_column_pos: Option<usize>,
}

fn extract_projection_expr(
    projection_ids: Option<&[u64]>,
    column_ids: &[u64],
    column_fields: &[DuckdbField],
) -> ProjectionWithVirtualColumns {
    // If projection ids are empty, use column_ids.
    // See duckdb/src/planner/operator/logical_get.cpp#L168
    let (ids, has_projection_ids) = match projection_ids {
        Some(ids) => (ids, true),
        None => (column_ids, false),
    };

    let mut file_index_column_pos = None;
    let mut file_row_number_column_pos = None;
    let mut is_star = true;
    let mut real_column_count = 0;

    // DuckDB uses u64 as column indices but Rust uses usize
    for (column_pos, &column_id) in ids.iter().enumerate() {
        let column_id = if has_projection_ids {
            let column_id: usize = column_id.as_();
            column_ids[column_id]
        } else {
            column_id
        };

        if column_id == FILE_INDEX_COLUMN_IDX {
            file_index_column_pos = Some(column_pos);
            continue;
        }
        if column_id == FILE_ROW_NUMBER_COLUMN_IDX {
            file_row_number_column_pos = Some(column_pos);
            continue;
        }

        // In SELECT * DuckDB requests all columns from 0 to column_fields in
        // increasing order. After removing virtual columns, compare column_id
        // with (0..column_fields.len()) range.
        is_star &= column_id == real_column_count;
        real_column_count += 1;
    }
    // Duckdb can request less columns than there are in table i.e. [0, 1] with
    // 5 columns total.
    is_star &= real_column_count == column_fields.len() as u64;

    let select = if is_star {
        root()
    } else {
        let names = ids
            .iter()
            .map(|&column_id| {
                if has_projection_ids {
                    let column_id: usize = column_id.as_();
                    column_ids[column_id]
                } else {
                    column_id
                }
            })
            .filter(|&col_id| !is_virtual_column(col_id))
            .map(|column_id| {
                let column_id: usize = column_id.as_();
                Arc::from(column_fields[column_id].name.as_str())
            })
            .collect::<FieldNames>();

        select(names, root())
    };

    // file_index column will be filled later when exporting the chunk.
    let projection = if file_row_number_column_pos.is_some() {
        // row_idx will be moved to correct position in scan(), prepend here
        let row_idx_struct = pack([("file_row_number", row_idx())], false.into());
        merge([row_idx_struct, select])
    } else {
        select
    };

    ProjectionWithVirtualColumns {
        projection,
        file_index_column_pos,
        file_row_number_column_pos,
    }
}

struct FilterWithVirtualColumns {
    filter: Option<Expression>,
    row_selection: Selection,
    row_range: Option<Range<u64>>,
    file_selection: Selection,
    file_range: Option<Range<u64>>,
}

/// Creates a table filter expression, row selection, and row range from the table filter set,
/// column metadata, additional filter expressions, and the top-level DType.
fn extract_table_filter_expr(
    table_filter_set: Option<&TableFilterSetRef>,
    column_ids: &[u64],
    column_fields: &[DuckdbField],
    additional_filters: &[Expression],
    dtype: &DType,
) -> VortexResult<FilterWithVirtualColumns> {
    let mut table_filter_exprs: HashSet<Expression> = if let Some(filter) = table_filter_set {
        filter
            .into_iter()
            .filter(|(idx, _)| {
                let idx_u: usize = idx.as_();
                !is_virtual_column(column_ids[idx_u])
            })
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

    let mut file_selection = Selection::All;
    let mut row_selection = Selection::All;
    let mut row_range = None;
    let mut file_range = None;
    if let Some(filter) = table_filter_set {
        for (idx, expression) in filter.into_iter() {
            let idx: usize = idx.as_();
            if column_ids[idx] == FILE_ROW_NUMBER_COLUMN_IDX {
                (row_selection, row_range) = try_from_virtual_column_filter(expression)?;
            }
            if column_ids[idx] == FILE_INDEX_COLUMN_IDX {
                (file_selection, file_range) = try_from_virtual_column_filter(expression)?;
            }
        }
    };

    let out = FilterWithVirtualColumns {
        filter: and_collect(table_filter_exprs),
        row_selection,
        row_range,
        file_selection,
        file_range,
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;

    use vortex::dtype::DType;
    use vortex::expr::merge;
    use vortex::expr::pack;
    use vortex::expr::root;
    use vortex::layout::layouts::row_idx::row_idx;

    use super::progress;
    use crate::datasource::DuckdbField;
    use crate::datasource::FILE_INDEX_COLUMN_IDX;
    use crate::datasource::FILE_ROW_NUMBER_COLUMN_IDX;
    use crate::datasource::extract_projection_expr;
    use crate::duckdb::LogicalType;

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

    #[test]
    fn test_select_star() {
        let ids = [0, 1, 2];
        let fields = [
            DuckdbField {
                name: "".to_owned(),
                logical_type: LogicalType::null(),
                dtype: DType::Null,
            },
            DuckdbField {
                name: "".to_owned(),
                logical_type: LogicalType::null(),
                dtype: DType::Null,
            },
            DuckdbField {
                name: "".to_owned(),
                logical_type: LogicalType::null(),
                dtype: DType::Null,
            },
        ];

        assert_eq!(
            extract_projection_expr(None, &ids, &fields).projection,
            root()
        );

        let ids = [FILE_ROW_NUMBER_COLUMN_IDX, 0, 1, FILE_INDEX_COLUMN_IDX, 2];
        let exprs = extract_projection_expr(None, &ids, &fields);
        let row_idx_struct = pack([("file_row_number", row_idx())], false.into());
        let root_with_virtual_cols = merge([row_idx_struct, root()]);

        assert_eq!(exprs.projection, root_with_virtual_cols);
        assert_eq!(exprs.file_index_column_pos, Some(3));
        assert_eq!(exprs.file_row_number_column_pos, Some(0));

        // projections can't be set in SELECT *.
        assert_ne!(
            extract_projection_expr(Some(&[0, 1]), &ids, &fields).projection,
            root()
        );

        let ids = [0, 1];
        assert_ne!(
            extract_projection_expr(None, &ids, &fields).projection,
            root()
        );

        let ids = [0, 2, 2];
        assert_ne!(
            extract_projection_expr(None, &ids, &fields).projection,
            root()
        );

        let ids = [2, 1, 0];
        assert_ne!(
            extract_projection_expr(None, &ids, &fields).projection,
            root()
        );
    }
}
