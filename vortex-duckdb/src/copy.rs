// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_fs::OpenOptions;
use futures::SinkExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::channel::mpsc::Sender;
use parking_lot::Mutex;
use static_assertions::assert_impl_all;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability::NonNullable;
use vortex::dtype::Nullability::Nullable;
use vortex::dtype::StructFields;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteSummary;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::io::runtime::current::CurrentThreadWorkerPool;
use vortex::io::session::RuntimeSessionExt;

use crate::RUNTIME;
use crate::SESSION;
use crate::convert::FromLogicalType;
use crate::convert::data_chunk_to_vortex;
use crate::duckdb::DataChunkRef;
use crate::duckdb::LogicalTypeRef;

#[derive(Clone)]
pub struct CopyFunctionBind {
    dtype: DType,
    fields: StructFields,
}
assert_impl_all!(CopyFunctionBind: Send, Clone);

/// Write to a file has two phases, writing data chunks and then closing the file.
/// We use a spawned tokio task to actually compress arrays and write it to disk.
/// Each chunk is pushed into the sink and read from the task.
/// Once finished we can close all sinks and then the task can be awaited and the file
/// flushed to disk.
pub struct CopyFunctionGlobal {
    write_task: Mutex<Option<Task<VortexResult<WriteSummary>>>>,
    sink: Option<Sender<VortexResult<ArrayRef>>>,
    // Pool of background workers helping to drive the write task.
    // Note that this is optional and without it, we would only drive the task when DuckDB calls
    // into us, and we call `RUNTIME.block_on`.
    // TODO(myrrc): we should rely only on host threads, remove this
    #[expect(dead_code)]
    worker_pool: CurrentThreadWorkerPool,
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

pub fn copy_to_sink(
    bind_data: &CopyFunctionBind,
    init_global: &CopyFunctionGlobal,
    chunk: &mut DataChunkRef,
) -> VortexResult<()> {
    let chunk = data_chunk_to_vortex(bind_data.fields.names(), chunk);
    let mut sink = init_global
        .sink
        .as_ref()
        .ok_or_else(|| vortex_err!("sink closed early"))?
        .clone();
    RUNTIME
        .block_on(sink.send(chunk))
        .map_err(|e| vortex_err!("send error {e}"))?;
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
        task.await?;
        Ok(())
    })
}

pub fn copy_to_initialize_global(
    bind_data: &CopyFunctionBind,
    file_path: String,
) -> VortexResult<CopyFunctionGlobal> {
    // The channel size 32 was chosen arbitrarily.
    let (sink, rx) = mpsc::channel(32);
    let array_stream = ArrayStreamAdapter::new(bind_data.dtype.clone(), rx.into_stream());

    let handle = SESSION.handle();

    let write_task = handle.spawn(async move {
        let writer = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(file_path)
            .await?;
        SESSION.write_options().write(writer, array_stream).await
    });

    let worker_pool = RUNTIME.new_pool();
    worker_pool.set_workers_to_available_parallelism();
    Ok(CopyFunctionGlobal {
        worker_pool,
        write_task: Mutex::new(Some(write_task)),
        sink: Some(sink),
    })
}
