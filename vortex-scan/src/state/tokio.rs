// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A scan driver designed for engines that use a Tokio runtime for orchestrating work, including
//! CPU-bound tasks. Let's not beat around the bush, this is targeted at DataFusion :)

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tokio::runtime::Handle;
use vortex_array::ArrayRef;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_error::VortexResult;

use crate::state::{Scan2, ScanTask, Scheduler, TaskSpawner};

impl Scan2 {
    pub fn into_tokio_steam(self, handle: Handle) -> impl ArrayStream + Send {
        let spawner: Box<dyn TaskSpawner> = Box::new(handle);
        let dtype = self.ctx.dtype.clone();
        ArrayStreamAdapter::new(dtype, self.into_scheduler(spawner))
    }
}

impl TaskSpawner for Handle {
    fn spawn_task(&self, task: Box<dyn ScanTask>) {
        // NOTE(ngates): we make an explicit choice not to spawn_blocking here as this is the
        //  compute model for DataFusion.
        // NOTE(ngates): we can safely drop the join handle since we don't use its result, the
        //  spawned task will continue running in the background.
        drop(Handle::spawn(self, async move { task.execute() }));
    }
}

impl Stream for Scheduler {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let pending = match self.make_progress_with_cx(cx) {
                Poll::Ready(Ok(())) => false,
                Poll::Ready(Err(e)) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Pending => true,
            };

            if let Some(array) = self.output_buffer.pop_front() {
                return Poll::Ready(Some(array));
            }

            if self.finished {
                return Poll::Ready(None);
            }

            if pending {
                return Poll::Pending;
            }
        }
    }
}
