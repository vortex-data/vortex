// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::iter;

use futures::SinkExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::channel::mpsc::Sender;
use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::dtype::DType;
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
use crate::convert::data_chunk_to_vortex;
use crate::convert::from_duckdb_table;
use crate::duckdb::ClientContextRef;
use crate::duckdb::CopyFunction;
use crate::duckdb::DataChunkRef;
use crate::duckdb::DuckDbFsWriter;
use crate::duckdb::LogicalTypeRef;

#[derive(Debug)]
pub struct VortexCopyFunction;

pub struct BindData {
    dtype: DType,
    fields: StructFields,
}

/// Write to a file has two phases, writing data chunks and then closing the file.
/// We use a spawned tokio task to actually compress arrays are write it to disk.
/// Each chunk is pushed into the sink and read from the task.
/// Once finished we can close all sinks and then the task can be awaited and the file
/// flushed to disk.
pub struct GlobalState {
    write_task: Mutex<Option<Task<VortexResult<WriteSummary>>>>,
    sink: Option<Sender<VortexResult<ArrayRef>>>,
    // Pool of background workers helping to drive the write task.
    // Note that this is optional and without it, we would only drive the task when DuckDB calls
    // into us, and we call `RUNTIME.block_on`.
    #[allow(dead_code)]
    worker_pool: CurrentThreadWorkerPool,
}

impl CopyFunction for VortexCopyFunction {
    type BindData = BindData;
    type GlobalState = GlobalState;
    type LocalState = ();

    fn bind(
        column_names: Vec<String>,
        column_types: Vec<&LogicalTypeRef>,
    ) -> VortexResult<Self::BindData> {
        let fields = from_duckdb_table(
            column_names
                .iter()
                .zip(column_types)
                .zip(iter::repeat(Nullable))
                .map(|((name, type_), null)| (name, type_, null)),
        )?;

        Ok(BindData {
            dtype: DType::Struct(fields.clone(), NonNullable),
            fields,
        })
    }

    fn copy_to_sink(
        bind_data: &Self::BindData,
        init_global: &Self::GlobalState,
        _init_local: &mut Self::LocalState,
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

    fn copy_to_finalize(
        _bind_data: &Self::BindData,
        init_global: &mut Self::GlobalState,
    ) -> VortexResult<()> {
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

    fn init_global(
        client_context: &ClientContextRef,
        bind_data: &Self::BindData,
        file_path: String,
    ) -> VortexResult<Self::GlobalState> {
        // The channel size 32 was chosen arbitrarily.
        let (sink, rx) = mpsc::channel(32);
        let array_stream = ArrayStreamAdapter::new(bind_data.dtype.clone(), rx.into_stream());

        let handle = SESSION.handle();
        // SAFETY: The ClientContext is owned by the Connection and lives for the duration of
        // query execution. DuckDB keeps the connection alive while this copy function runs.
        let ctx = unsafe { client_context.erase_lifetime() };

        // Use DuckDB FS exclusively to match the DuckDB client context configuration.
        let writer = DuckDbFsWriter::new(ctx, &file_path)
            .map_err(|e| vortex_err!("Failed to create DuckDB FS writer for {file_path}: {e}"))?;

        let write_task =
            handle.spawn(async move { SESSION.write_options().write(writer, array_stream).await });

        let worker_pool = RUNTIME.new_pool();
        worker_pool.set_workers_to_available_parallelism();

        Ok(GlobalState {
            worker_pool,
            write_task: Mutex::new(Some(write_task)),
            sink: Some(sink),
        })
    }

    fn init_local(_global: &Self::BindData) -> VortexResult<Self::LocalState> {
        Ok(())
    }
}
