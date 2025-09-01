// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::iter;
use std::sync::{Arc, LazyLock};

use tokio::fs::File;
use tokio::runtime::{self, Handle, Runtime};
use tokio::sync::mpsc;
use tokio::sync::mpsc::Sender;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use vortex::ArrayRef;
use vortex::dtype::Nullability::{NonNullable, Nullable};
use vortex::dtype::{DType, StructFields};
use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex::stream::ArrayStreamAdapter;
use vortex_file::{VortexWriteOptions, WriteStrategyBuilder};

use crate::convert::{data_chunk_to_arrow, from_duckdb_table};
use crate::duckdb::{CopyFunction, DataChunk, LogicalType};

#[derive(Debug)]
pub struct VortexCopyFunction;

pub struct BindData {
    dtype: DType,
    fields: StructFields,
}

static COPY_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .vortex_expect("Cannot start runtime")
});

/// Write to a file has two phases, writing data chunks and then closing the file.
/// We use a spawned tokio task to actually compress arrays are write it to disk.
/// Each chunk is pushed into the sink and read from the task.
/// Once finished we can close all sinks and then the task can be awaited and the file
/// flushed to disk.
pub struct GlobalState {
    write_task: Option<JoinHandle<VortexResult<File>>>,
    sink: Option<Sender<VortexResult<ArrayRef>>>,
}

impl CopyFunction for VortexCopyFunction {
    type BindData = BindData;
    type GlobalState = GlobalState;
    type LocalState = ();

    fn bind(
        column_names: Vec<String>,
        column_types: Vec<LogicalType>,
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
        init_global: &mut Self::GlobalState,
        _init_local: &mut Self::LocalState,
        chunk: &mut DataChunk,
    ) -> VortexResult<()> {
        let chunk = data_chunk_to_arrow(bind_data.fields.names(), chunk);
        COPY_RUNTIME.block_on(async {
            init_global
                .sink
                .as_ref()
                .vortex_expect("sink closed early")
                .send(chunk)
                .await
                .map_err(|e| vortex_err!("send error {}", e.to_string()))
        })?;

        Ok(())
    }

    fn copy_to_finalize(
        _bind_data: &Self::BindData,
        init_global: &mut Self::GlobalState,
    ) -> VortexResult<()> {
        COPY_RUNTIME.block_on(async {
            if let Some(sink) = init_global.sink.take() {
                drop(sink)
            }
            let file = init_global
                .write_task
                .take()
                .vortex_expect("no file to close")
                .await??;
            file.sync_all().await?;
            Ok(())
        })
    }

    fn init_global(
        bind_data: &Self::BindData,
        file_path: String,
    ) -> VortexResult<Self::GlobalState> {
        // The channel size 32 was chosen arbitrarily.
        let (sink, rx) = mpsc::channel(32);
        let array_stream =
            ArrayStreamAdapter::new(bind_data.dtype.clone(), ReceiverStream::new(rx));

        let writer = COPY_RUNTIME.spawn(async move {
            let file = File::create(file_path).await?;
            VortexWriteOptions::default()
                .with_strategy(
                    WriteStrategyBuilder::new()
                        .with_executor(Arc::new(Handle::current()))
                        .build(),
                )
                .write(file, array_stream)
                .await
        });

        Ok(GlobalState {
            write_task: Some(writer),
            sink: Some(sink),
        })
    }

    fn init_local(_global: &Self::BindData) -> VortexResult<Self::LocalState> {
        Ok(())
    }
}
