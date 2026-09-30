// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_utils::parallelism::get_available_parallelism;

use crate::scan::scan_builder::ScanBuilder;
use crate::scan::v2::ScanFile;
use crate::scan::v2::prepare;

/// Returns a stream that prepares the scan over `file` on first poll and spawns its split tasks.
///
/// The replacement for [`ScanBuilder::into_stream`].
pub fn into_stream<A: 'static + Send>(
    builder: ScanBuilder<A>,
    file: ScanFile,
) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
    Ok(LazyScanStream {
        state: State::Builder(Some(Box::new((builder, file)))),
    })
}

type Tasks<A> = Vec<BoxFuture<'static, VortexResult<Vec<A>>>>;

enum State<A: 'static + Send> {
    Builder(Option<Box<(ScanBuilder<A>, ScanFile)>>),
    Preparing {
        ordered: bool,
        concurrency: usize,
        handle: Handle,
        task: Task<VortexResult<Tasks<A>>>,
    },
    Stream(BoxStream<'static, VortexResult<A>>),
    Error(Option<VortexError>),
}

struct LazyScanStream<A: 'static + Send> {
    state: State<A>,
}

impl<A: 'static + Send> Unpin for LazyScanStream<A> {}

impl<A: 'static + Send> Stream for LazyScanStream<A> {
    type Item = VortexResult<A>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                State::Builder(builder) => {
                    let (builder, file) = *builder.take().vortex_expect("polled after completion");
                    let ordered = builder.ordered();
                    let num_workers = get_available_parallelism().unwrap_or(1);
                    let concurrency = builder.concurrency() * num_workers;
                    let handle = builder.session().handle();
                    let task = handle.spawn_cpu(move || {
                        prepare(builder, file).and_then(|s| s.execute_batches(None))
                    });
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
                        let stream = futures::stream::iter(tasks).map(move |t| handle.spawn(t));
                        let stream = if *ordered {
                            stream.buffered(concurrency).boxed()
                        } else {
                            stream.buffer_unordered(concurrency).boxed()
                        };
                        // A task returns one batch per projection split of its filter split.
                        self.state = State::Stream(
                            stream
                                .flat_map(|batches| {
                                    futures::stream::iter(match batches {
                                        Ok(batches) => batches.into_iter().map(Ok).collect(),
                                        Err(err) => vec![Err(err)],
                                    })
                                })
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
