// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::future::Either;
use futures::stream::BoxStream;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_utils::parallelism::get_available_parallelism;

use crate::scan::scan_builder;
use crate::scan::v2::ScanBuilder;
use crate::scan::v2::ScanFile;
use crate::scan::v2::pruning::file_pruning_enabled;

/// Returns a stream that prepares the scan over `file` on first poll and runs its split tasks.
///
/// The replacement for the default
/// [`ScanBuilder::into_stream`](scan_builder::ScanBuilder::into_stream). Equivalent to copying
/// `builder` into a [`ScanBuilder`] and streaming that.
pub fn into_stream<A: 'static + Send>(
    builder: scan_builder::ScanBuilder<A>,
    file: ScanFile,
) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
    ScanBuilder::from_default(builder, file).into_stream()
}

type Tasks<A> = Vec<BoxFuture<'static, VortexResult<Vec<A>>>>;

enum State<A: 'static + Send> {
    Builder(Option<Box<ScanBuilder<A>>>),
    Preparing {
        ordered: bool,
        concurrency: usize,
        handle: Handle,
        task: Task<VortexResult<Tasks<A>>>,
    },
    Stream(BoxStream<'static, VortexResult<A>>),
    Error(Option<VortexError>),
}

/// Prepares a scan on first poll, then runs its split tasks and yields their batches.
pub(super) struct LazyScanStream<A: 'static + Send> {
    state: State<A>,
}

impl<A: 'static + Send> LazyScanStream<A> {
    pub(super) fn new(builder: ScanBuilder<A>) -> Self {
        Self {
            state: State::Builder(Some(Box::new(builder))),
        }
    }
}

impl<A: 'static + Send> Unpin for LazyScanStream<A> {}

impl<A: 'static + Send> Stream for LazyScanStream<A> {
    type Item = VortexResult<A>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                State::Builder(builder) => {
                    let builder = builder.take().vortex_expect("polled after completion");
                    let ordered = builder.ordered();
                    let num_workers = get_available_parallelism().unwrap_or(1);
                    let concurrency = builder.concurrency() * num_workers;
                    let handle = builder.session().handle();
                    let task = if file_pruning_enabled() {
                        let prepared = handle.spawn_cpu(move || builder.prepare());
                        handle.spawn(
                            async move { prepared.await?.execute_batches_pruned(None).await },
                        )
                    } else {
                        handle.spawn_cpu(move || {
                            builder
                                .prepare()
                                .and_then(|scan| scan.execute_batches(None))
                        })
                    };
                    self.state = State::Preparing {
                        ordered,
                        concurrency,
                        handle,
                        task,
                    };
                }
                State::Preparing {
                    ordered,
                    concurrency,
                    handle,
                    task,
                } => match ready!(Pin::new(task).poll(cx)) {
                    Ok(tasks) => {
                        let concurrency = *concurrency;
                        let handle = handle.clone();
                        // A single split already runs on the polling worker. Spawning it adds
                        // a scheduler handoff without exposing any more parallelism.
                        let spawn = tasks.len() > 1 && concurrency > 1;
                        let stream = futures::stream::iter(tasks).map(move |task| {
                            if spawn {
                                Either::Right(handle.spawn(task))
                            } else {
                                Either::Left(task)
                            }
                        });
                        let stream = if *ordered {
                            stream.buffered(concurrency).boxed()
                        } else {
                            stream.buffer_unordered(concurrency).boxed()
                        };
                        // A task returns one batch per projection split of its filter split.
                        self.state = State::Stream(
                            stream
                                .map_ok(|batches| {
                                    futures::stream::iter(batches.into_iter().map(Ok))
                                })
                                .try_flatten()
                                .boxed(),
                        );
                    }
                    Err(err) => self.state = State::Error(Some(err)),
                },
                State::Stream(stream) => return stream.as_mut().poll_next(cx),
                State::Error(err) => return Poll::Ready(err.take().map(Err)),
            }
        }
    }
}
