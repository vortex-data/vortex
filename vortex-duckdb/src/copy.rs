// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_fs::OpenOptions;
use futures::SinkExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::channel::mpsc::Sender;
use object_store::ObjectStore;
use object_store::registry::ObjectStoreRegistry;
use parking_lot::Mutex;
use static_assertions::assert_impl_all;
use vortex::aggregate_fn::AggregateFnRef;
use vortex::aggregate_fn::AggregateFnVTableExt;
use vortex::aggregate_fn::EmptyOptions;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::nan_count::NanCount;
use vortex::aggregate_fn::fns::null_count::NullCount;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability::NonNullable;
use vortex::dtype::Nullability::Nullable;
use vortex::dtype::StructFields;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::stats::Precision;
use vortex::file::CompressedFieldSizes;
use vortex::file::FileStatistics;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteSummary;
use vortex::file::multi::parse_uri_or_path;
use vortex::io::VortexWrite;
use vortex::io::compat::Compat;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::io::session::RuntimeSessionExt;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;

use crate::REGISTRY;
use crate::RUNTIME;
use crate::SESSION;
use crate::convert::FromLogicalType;
use crate::convert::ToDuckDBScalar;
use crate::convert::data_chunk_to_vortex;
use crate::duckdb::DataChunkRef;
use crate::duckdb::LogicalTypeRef;
use crate::duckdb::Value;

#[derive(Clone)]
pub struct CopyFunctionBind {
    dtype: DType,
    fields: StructFields,
}
assert_impl_all!(CopyFunctionBind: Send, Clone);

/// The compressed field sizes and the statistics columns are computed once here rather than per
/// column, since DuckDB queries statistics one column at a time.
struct FinishedWrite {
    summary: WriteSummary,
    column_sizes: Option<CompressedFieldSizes>,
    /// Indices into the footer's [`FileStatistics`] of the entries reported as statistics columns.
    stats_columns: Vec<usize>,
}

/// Write to a file has two phases, writing data chunks and then closing the file.
/// We use a spawned tokio task to actually compress arrays and write it to disk.
/// Each chunk is pushed into the sink and read from the task.
/// Once finished we can close all sinks and then the task can be awaited and the file
/// flushed to disk.
pub struct CopyFunctionGlobal {
    write_task: Mutex<Option<Task<VortexResult<WriteSummary>>>>,
    finished: Mutex<Option<FinishedWrite>>,
    sink: Option<Sender<VortexResult<ArrayRef>>>,
}
assert_impl_all!(CopyFunctionGlobal: Send, Sync);

pub fn copy_to_bind(
    column_names: &[String],
    column_types: &[&LogicalTypeRef],
) -> VortexResult<CopyFunctionBind> {
    let fields: StructFields = column_names
        .iter()
        .zip(column_types)
        .map(|(name, type_)| {
            Ok((
                FieldName::from(name.as_ref()),
                DType::from_logical_type(type_, Nullable)?,
            ))
        })
        .collect::<VortexResult<StructFields>>()?;

    Ok(CopyFunctionBind {
        dtype: DType::Struct(fields.clone(), NonNullable),
        fields,
    })
}

fn push_to_writer(global: &CopyFunctionGlobal, array: ArrayRef) -> VortexResult<()> {
    let mut sink = global
        .sink
        .as_ref()
        .ok_or_else(|| vortex_err!("sink closed early"))?
        .clone();
    RUNTIME.block_on(async {
        // send may error with "receiver is gone" which isn't the real error
        if sink.send(Ok(array)).await.is_ok() {
            return Ok(());
        }
        let task = global.write_task.lock().take();
        if let Some(task) = task {
            // we can get the real error (i.e invalid path) from here
            task.await?;
        }
        vortex_bail!("Writer stopped before all data was written")
    })
}

pub fn copy_to_sink(
    bind_data: &CopyFunctionBind,
    init_global: &CopyFunctionGlobal,
    chunk: &mut DataChunkRef,
) -> VortexResult<()> {
    push_to_writer(
        init_global,
        data_chunk_to_vortex(bind_data.fields.names(), chunk)?,
    )
}

#[derive(Default)]
pub struct CopyPreparedBatch {
    arrays: Vec<ArrayRef>,
}

pub fn prepare_batch_push(
    bind: &CopyFunctionBind,
    batch: &mut CopyPreparedBatch,
    chunk: &DataChunkRef,
) -> VortexResult<()> {
    batch
        .arrays
        .push(data_chunk_to_vortex(bind.fields.names(), chunk)?);
    Ok(())
}

pub fn flush_batch(global: &CopyFunctionGlobal, batch: &CopyPreparedBatch) -> VortexResult<()> {
    for array in &batch.arrays {
        push_to_writer(global, array.clone())?;
    }
    Ok(())
}

pub fn copy_to_finalize(init_global: &mut CopyFunctionGlobal) -> VortexResult<()> {
    RUNTIME.block_on(async {
        if let Some(sink) = init_global.sink.take() {
            drop(sink)
        }
        let task = init_global
            .write_task
            .lock()
            .take()
            .vortex_expect("no file to close");
        // Keep the write summary (footer + size) so DuckLake can read per-file statistics back
        // without re-opening the file. Compute the compressed field sizes once, up front.
        let summary = task.await?;
        let column_sizes = summary.footer().compressed_field_sizes().ok();
        let stats_columns = summary
            .footer()
            .statistics()
            .map(leaf_stats_columns)
            .unwrap_or_default();
        *init_global.finished.lock() = Some(FinishedWrite {
            summary,
            column_sizes,
            stats_columns,
        });
        Ok(())
    })
}

/// File-level statistics of the written Vortex file, for the WRITTEN_FILE_STATISTICS return path.
pub(crate) struct WrittenFileStats {
    pub row_count: u64,
    pub file_size_bytes: u64,
    pub footer_size_bytes: u64,
    pub num_columns: usize,
}

/// Per-column statistics of the written Vortex file. `min`/`max` are DuckDB values converted from
/// the Vortex scalar; every field is optional and omitted when the statistic is not available.
pub(crate) struct WrittenColumnStats {
    /// The column's path in DuckLake's key format (see [`ducklake_column_key`]).
    pub column_key: String,
    pub min: Option<Value>,
    pub max: Option<Value>,
    pub null_count: Option<u64>,
    pub num_values: u64,
    pub column_size_bytes: Option<u64>,
    pub has_nan: Option<bool>,
}

/// Read file-level statistics back from the finished write. `None` before finalize.
pub(crate) fn written_file_stats(global: &CopyFunctionGlobal) -> Option<WrittenFileStats> {
    let guard = global.finished.lock();
    let finished = guard.as_ref()?;
    Some(file_stats_from_summary(
        &finished.summary,
        finished.stats_columns.len(),
    ))
}

/// Read per-column statistics for `column_index` from the finished write. `Ok(None)` if the file is
/// not finalized.
pub(crate) fn written_column_stats(
    global: &CopyFunctionGlobal,
    column_index: usize,
) -> VortexResult<Option<WrittenColumnStats>> {
    let guard = global.finished.lock();
    let Some(finished) = guard.as_ref() else {
        return Ok(None);
    };
    column_stats_from_summary(
        &finished.summary,
        &finished.stats_columns,
        column_index,
        finished.column_sizes.as_ref(),
    )
    .map(Some)
}

/// The indices of the leaf entries of `file_stats`, which are the columns statistics are reported
/// for. Like parquet, which reports one entry per leaf column writer, struct columns get no entry of
/// their own; nested fields are reported at their full path. Lists and maps are opaque leaves of the
/// statistics layout, so they are reported at their own path rather than per element.
fn leaf_stats_columns(file_stats: &FileStatistics) -> Vec<usize> {
    file_stats
        .dtypes()
        .iter()
        .enumerate()
        .filter(|(_, dtype)| !dtype.is_struct())
        .map(|(index, _)| index)
        .collect()
}

/// DuckLake keys column statistics by a quoted, dot-separated path (see
/// `DuckLakeUtil::ParseQuotedList`). This matches the parquet writer, which quotes each name with
/// `KeywordHelper::WriteQuoted(name, '"')`, escaping embedded quotes by doubling them. A list
/// element is named `element`, as in parquet's list encoding.
fn ducklake_column_key(path: &FieldPath) -> String {
    path.parts()
        .iter()
        .map(|part| {
            let name = part.as_name().unwrap_or("element");
            format!("\"{}\"", name.replace('"', "\"\""))
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn file_stats_from_summary(summary: &WriteSummary, num_columns: usize) -> WrittenFileStats {
    WrittenFileStats {
        row_count: summary.row_count(),
        file_size_bytes: summary.size(),
        // Vortex has no separate footer-size hint; 0 means "read the footer normally".
        footer_size_bytes: 0,
        num_columns,
    }
}

/// Statistics for the `column_index`-th statistics column of a finished write, where
/// `stats_columns` are the [leaf entries][leaf_stats_columns] of its footer statistics and
/// `column_sizes` its compressed field sizes.
fn column_stats_from_summary(
    summary: &WriteSummary,
    stats_columns: &[usize],
    column_index: usize,
    column_sizes: Option<&CompressedFieldSizes>,
) -> VortexResult<WrittenColumnStats> {
    let file_stats = summary
        .footer()
        .statistics()
        .ok_or_else(|| vortex_err!("written file has no statistics"))?;
    let stats_index = *stats_columns.get(column_index).ok_or_else(|| {
        vortex_err!(
            "column index {column_index} out of range for {} statistics columns",
            stats_columns.len()
        )
    })?;
    let (aggregates, dtype) = file_stats.get(stats_index);
    let path = &file_stats.paths()[stats_index];
    let value =
        |aggregate_fn: AggregateFnRef| aggregates.get(&aggregate_fn).and_then(Scalar::into_value);

    Ok(WrittenColumnStats {
        column_key: ducklake_column_key(path),
        min: exact_scalar_to_duckdb(value(Min.bind(NumericalAggregateOpts::skip_nans())), dtype)?,
        max: exact_scalar_to_duckdb(value(Max.bind(NumericalAggregateOpts::skip_nans())), dtype)?,
        null_count: exact_u64(value(NullCount.bind(EmptyOptions))),
        // NaNCount is exact only for float columns, so this is emitted just for them (as in parquet).
        has_nan: exact_u64(value(NanCount.bind(EmptyOptions))).map(|count| count > 0),
        num_values: summary.row_count(),
        // On-disk compressed size; excludes bytes not attributable to a column (e.g. struct validity).
        column_size_bytes: column_sizes.and_then(|sizes| sizes.get(path)),
    })
}

/// Convert an exact scalar statistic to a DuckDB value, propagating a conversion failure rather than
/// dropping it. `Ok(None)` when the statistic is not exactly known.
fn exact_scalar_to_duckdb(
    stat: Precision<ScalarValue>,
    dtype: &DType,
) -> VortexResult<Option<Value>> {
    match stat {
        Precision::Exact(value) => Ok(Some(
            Scalar::try_new(dtype.clone(), Some(value))?.try_to_duckdb_scalar()?,
        )),
        _ => Ok(None),
    }
}

/// Extract an exact `u64` statistic (e.g. a count), or `None` if not exactly known.
fn exact_u64(stat: Precision<ScalarValue>) -> Option<u64> {
    match stat {
        Precision::Exact(value) => value.as_primitive().as_u64(),
        _ => None,
    }
}

pub fn copy_to_initialize_global(
    bind_data: &CopyFunctionBind,
    file_path: String,
) -> VortexResult<CopyFunctionGlobal> {
    // The channel size 32 was chosen arbitrarily.
    let (sink, rx) = mpsc::channel(32);
    let array_stream = ArrayStreamAdapter::new(bind_data.dtype.clone(), rx.into_stream());

    let handle = SESSION.handle();

    let url = parse_uri_or_path(&file_path)?;
    let write_task = if url.scheme() == "file" {
        handle.spawn(async move {
            let mut writer = OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(file_path)
                .await?;
            let summary = SESSION
                .write_options()
                .write(&mut writer, array_stream)
                .await?;
            writer.shutdown().await?;
            Ok(summary)
        })
    } else {
        let (object_store, path) = REGISTRY.resolve(&url)?;
        let object_store = Arc::new(Compat::new(object_store)) as Arc<dyn ObjectStore>;
        handle.spawn(async move {
            let mut writer = ObjectStoreWrite::new(object_store, &path).await?;
            let summary = SESSION
                .write_options()
                .write(&mut writer, array_stream)
                .await?;
            writer.shutdown().await?;
            Ok(summary)
        })
    };

    Ok(CopyFunctionGlobal {
        write_task: Mutex::new(Some(write_task)),
        finished: Mutex::new(None),
        sink: Some(sink),
    })
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::ByteBufferMut;
    use vortex::buffer::buffer;
    use vortex::expr::stats::Stat;

    use super::*;

    fn pruning_aggregate_fns() -> Vec<AggregateFnRef> {
        [Stat::Min, Stat::Max, Stat::NullCount, Stat::NaNCount]
            .into_iter()
            .filter_map(|stat| stat.aggregate_fn())
            .collect()
    }

    /// Writes a one-column file and returns its summary, with `file_statistics` controlling which
    /// statistics the footer carries (empty means none at all).
    fn write_summary(file_statistics: Vec<AggregateFnRef>) -> WriteSummary {
        RUNTIME.block_on(async {
            let array = StructArray::from_fields(&[("i", buffer![1u32, 2, 3].into_array())])
                .unwrap()
                .into_array();
            let mut buf = ByteBufferMut::empty();
            let mut writer = SESSION
                .write_options()
                .with_file_statistics(file_statistics)
                .writer(&mut buf, array.dtype().clone());
            writer.push(array).await.unwrap();
            writer.finish().await.unwrap()
        })
    }

    /// Statistics for the `column_index`-th statistics column of `summary`, resolved the same way
    /// `copy_to_finalize` and [`written_column_stats`] do.
    fn column_stats(
        summary: &WriteSummary,
        column_index: usize,
    ) -> VortexResult<WrittenColumnStats> {
        let stats_columns = summary
            .footer()
            .statistics()
            .map(leaf_stats_columns)
            .unwrap_or_default();
        let column_sizes = summary.footer().compressed_field_sizes()?;
        column_stats_from_summary(summary, &stats_columns, column_index, Some(&column_sizes))
    }

    #[test]
    fn column_stats_out_of_range_is_an_error() {
        let summary = write_summary(pruning_aggregate_fns());
        assert!(column_stats(&summary, 0).is_ok());
        assert!(column_stats(&summary, 1).is_err());
    }

    #[test]
    fn column_stats_without_file_statistics_is_an_error() {
        let summary = write_summary(vec![]);
        assert!(column_stats(&summary, 0).is_err());
    }

    #[test]
    fn column_stats_reports_nested_leaves_by_quoted_path() -> VortexResult<()> {
        // A nullable struct `s` contributes `s.b` and a trailing own entry to the post-order stats
        // layout; only the leaves are statistics columns, keyed by their full quoted path.
        let summary = RUNTIME.block_on(async {
            let b = buffer![10i32, 20, 30].into_array();
            let inner = StructArray::new(["b"].into(), [b], 3, Validity::AllValid).into_array();
            let c = buffer![7u32, 8, 9].into_array();
            let outer =
                StructArray::new(["s", "c\"q"].into(), [inner, c], 3, Validity::NonNullable)
                    .into_array();

            let mut buf = ByteBufferMut::empty();
            let mut writer = SESSION
                .write_options()
                .with_file_statistics(pruning_aggregate_fns())
                .writer(&mut buf, outer.dtype().clone());
            writer.push(outer).await?;
            writer.finish().await
        })?;

        let file_stats = summary
            .footer()
            .statistics()
            .ok_or_else(|| vortex_err!("no statistics"))?;
        assert_eq!(leaf_stats_columns(file_stats).len(), 2);
        let nested = column_stats(&summary, 0)?;
        assert_eq!(nested.column_key, "\"s\".\"b\"");
        assert!(nested.min.is_some());
        assert!(nested.column_size_bytes.is_some_and(|size| size > 0));

        let top_level = column_stats(&summary, 1)?;
        assert_eq!(top_level.column_key, "\"c\"\"q\"");
        assert!(top_level.min.is_some());
        assert!(top_level.max.is_some());

        assert!(column_stats(&summary, 2).is_err());
        Ok(())
    }
}
