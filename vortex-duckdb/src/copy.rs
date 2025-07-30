// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::task::noop_waker;
use futures::{FutureExt, pin_mut, stream};
use parking_lot::Mutex;
use std::fmt::Debug;
use std::iter;
use std::sync::Arc;
use std::task::{Context, Poll};
use futures::channel::mpsc;
use futures::channel::mpsc::Sender;
use tokio::fs::File;
use vortex::ArrayRef;
use vortex::dtype::Nullability::{NonNullable, Nullable};
use vortex::dtype::{DType, StructFields};
use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex::iter::ArrayIteratorAdapter;
use vortex::stream::ArrayStreamAdapter;
use vortex_file::VortexWriteOptions;

use crate::RUNTIME;
use crate::convert::{data_chunk_to_array, from_duckdb_table};
use crate::duckdb::{CopyFunction, DataChunk, LogicalType};

#[derive(Debug)]
pub struct VortexCopyFunction;

pub struct BindData {
    dtype: DType,
    fields: StructFields,
}

/// Write to a file has two phases, writing data chunks and then closing the file.
/// We don't currently support parallel compression in this model, so we set up the writer in
/// the global state and drive it each time a local thread receives a chunk of data.
pub struct GlobalState {
    sink: Option<Sender<VortexResult<ArrayRef>>>,
    writer: Mutex<Option<BoxFuture<'static, VortexResult<()>>>>,
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

    /// Invoked on a local worker thread to copy data to the sink.
    fn copy_to_sink(
        bind_data: &Self::BindData,
        init_global: &mut Self::GlobalState,
        _init_local: &mut Self::LocalState,
        chunk: &mut DataChunk,
    ) -> VortexResult<()> {
        // We push the chunk into the sink, then drive the writer as much as possible.
        init_global
            .sink
            .as_ref()
            .vortex_expect("sink closed")
            .send(data_chunk_to_array(bind_data.fields.names(), chunk))
            .map_err(|e| vortex_err!("send error {}", e.to_string()))?;

        // We don't care about waking up the writer task, since it isn't running itself. Therefore,
        // we can use a no-op waker to drive until pending.
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut guard = init_global.writer.lock();
        if let Some(mut writer) = guard.take() {
            loop {
                match writer.poll_unpin(&mut context) {
                    Poll::Ready(Ok(())) => {
                        unreachable!(
                            "Ther writer task should not complete until the sink is closed"
                        );
                    }
                    Poll::Ready(Err(e)) => {
                        // Bail out if the writer task failed.
                        return Err(e);
                    }
                    Poll::Pending => {
                        // If the writer is pending, put the writer back in the guard and exit.
                        *guard = Some(writer);
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    fn copy_to_finalize(
        _bind_data: &Self::BindData,
        init_global: &mut Self::GlobalState,
    ) -> VortexResult<()> {
        // In the finalize phase, we close the sink and wait for the writer task to complete.
        drop(init_global.sink.take());

        // Now we drive the future to completion.
        let writer = init_global
            .writer
            .lock()
            .take()
            .ok_or_else(|| vortex_err!("writer task already failed"))?;
        block_on(writer)?;

        RUNTIME.block_on(async {
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
        let (send, recv) = mpsc::channel(32);

        let array_stream =
            ArrayStreamAdapter::new(bind_data.dtype.clone(), recv);

        let writer = async move {
            let file = File::create(&file_path).await?;
            let buffers = VortexWriteOptions::default()
                .write_tokio(array_stream);
        }

        let writer = RUNTIME.spawn(async move {
            let file = File::create(file_path).await?;
            VortexWriteOptions::default()
                .write_tokio(file, array_stream)
                .await
        });

        Ok(GlobalState {
            sink: Some(send),
            writer: Default::default(),
        })
    }

    fn init_local(_global: &Self::BindData) -> VortexResult<Self::LocalState> {
        Ok(())
    }
}
