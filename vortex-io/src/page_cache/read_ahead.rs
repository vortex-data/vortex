// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fetches the pages that follow a read before they are asked for, within a bound.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use tokio::sync::Semaphore;
use vortex_utils::aliases::hash_map::HashMap;

use crate::runtime::Handle;
use crate::runtime::Task;

/// The most objects whose reads the read-ahead follows. Past it, it forgets the object read
/// least recently.
const MAX_TRACKED_OBJECTS: usize = 128;

/// Follows the reads of each object and fetches the pages after them.
///
/// Each page fetch is one task, and at most `max_in_flight` run at the same time. A read that
/// does not continue from the reads before it cancels the fetches of its object, and dropping
/// the read-ahead cancels them all.
pub(super) struct ReadAhead {
    max_pages: u64,
    slots: Arc<Semaphore>,
    streams: Mutex<Streams>,
}

#[derive(Default)]
struct Streams {
    clock: u64,
    objects: HashMap<Arc<str>, ObjectStream>,
}

/// The reads of one object, and its page fetches that may still run.
struct ObjectStream {
    first_page: u64,
    last_page: u64,
    touched: u64,
    fetches: BTreeMap<u64, Fetch>,
}

/// A page fetch. Dropping it cancels the fetch.
struct Fetch {
    _task: Task<()>,
    done: Arc<AtomicBool>,
}

impl ReadAhead {
    /// Fetch at most `max_pages` pages after each read, with at most `max_in_flight` fetches
    /// at the same time.
    pub(super) fn new(max_pages: usize, max_in_flight: usize) -> Self {
        Self {
            max_pages: max_pages as u64,
            slots: Arc::new(Semaphore::new(max_in_flight)),
            streams: Mutex::new(Streams::default()),
        }
    }

    /// Follows a completed read of pages `first_page..=last_page` of an object with `page_count`
    /// pages, and fetches the pages after it on the tasks of `handle`.
    ///
    /// The `fetch` closure must not keep the owner of this read-ahead alive: dropping that owner
    /// cancels the fetches. A fetch never waits for a slot, so demand reads keep priority.
    pub(super) fn observe(
        &self,
        handle: &Handle,
        object_key: Arc<str>,
        first_page: u64,
        last_page: u64,
        page_count: u64,
        fetch: impl Fn(u64) -> BoxFuture<'static, ()> + Send + Sync + 'static,
    ) {
        if self.max_pages == 0 || first_page > last_page || last_page >= page_count {
            return;
        }

        let mut streams = self.streams.lock();
        streams.clock = streams.clock.wrapping_add(1);
        let touched = streams.clock;

        if !streams.objects.contains_key(&object_key)
            && streams.objects.len() >= MAX_TRACKED_OBJECTS
            && let Some(oldest) = streams
                .objects
                .iter()
                .min_by_key(|(_, stream)| stream.touched)
                .map(|(key, _)| Arc::clone(key))
        {
            streams.objects.remove(&oldest);
        }

        let stream = streams
            .objects
            .entry(object_key)
            .or_insert_with(|| ObjectStream {
                first_page,
                last_page,
                touched,
                fetches: BTreeMap::new(),
            });

        // A read that goes back, or that skips past what the read-ahead fetched, is a seek.
        let seek = first_page < stream.first_page
            || first_page
                > stream
                    .last_page
                    .saturating_add(self.max_pages.saturating_add(1));
        if seek {
            stream.fetches.clear();
            stream.first_page = first_page;
            stream.last_page = last_page;
        } else {
            stream.first_page = first_page;
            stream.last_page = stream.last_page.max(last_page);
        }
        stream.touched = touched;
        stream
            .fetches
            .retain(|_, fetch| !fetch.done.load(Ordering::Acquire));

        let end = last_page
            .saturating_add(1)
            .saturating_add(self.max_pages)
            .min(page_count);
        for page in last_page + 1..end {
            if stream.fetches.contains_key(&page) {
                continue;
            }

            let Ok(permit) = Arc::clone(&self.slots).try_acquire_owned() else {
                break;
            };

            let future = fetch(page);
            let done = Arc::new(AtomicBool::new(false));
            let task = {
                let done = Arc::clone(&done);
                handle.spawn(async move {
                    let _permit = permit;
                    future.await;
                    done.store(true, Ordering::Release);
                })
            };
            stream.fetches.insert(page, Fetch { _task: task, done });
        }
    }
}

#[cfg(test)]
#[cfg(feature = "tokio")]
mod tests {
    use std::sync::Arc;

    use futures::FutureExt;
    use futures::future::BoxFuture;
    use tokio::sync::mpsc;

    use super::ReadAhead;
    use crate::runtime::tokio::TokioRuntime;

    /// Reports its page when the fetch that holds it is dropped.
    struct Cancelled(u64, mpsc::UnboundedSender<u64>);

    impl Drop for Cancelled {
        fn drop(&mut self) {
            self.1.send(self.0).ok();
        }
    }

    /// A fetch that reports its page when it starts and runs until it is cancelled.
    fn pending_fetch(
        started: mpsc::UnboundedSender<u64>,
        cancelled: mpsc::UnboundedSender<u64>,
    ) -> impl Fn(u64) -> BoxFuture<'static, ()> + Send + Sync + 'static {
        move |page| {
            let started = started.clone();
            let cancelled = Cancelled(page, cancelled.clone());
            async move {
                let _cancelled = cancelled;
                started.send(page).ok();
                futures::future::pending::<()>().await;
            }
            .boxed()
        }
    }

    #[tokio::test]
    async fn read_ahead_is_bounded_and_cancelled_on_seek_and_drop() {
        let handle = TokioRuntime::current();
        let ahead = ReadAhead::new(4, 2);
        let (cancelled, mut cancels) = mpsc::unbounded_channel();

        let (started, mut starts) = mpsc::unbounded_channel();
        ahead.observe(
            &handle,
            Arc::from("object"),
            0,
            0,
            100,
            pending_fetch(started, cancelled.clone()),
        );
        assert_eq!(starts.recv().await, Some(1));
        assert_eq!(starts.recv().await, Some(2));
        assert!(starts.try_recv().is_err());

        ahead.observe(&handle, Arc::from("object"), 20, 20, 100, |_| {
            async {}.boxed()
        });
        let mut pages = [cancels.recv().await, cancels.recv().await];
        pages.sort();
        assert_eq!(pages, [Some(1), Some(2)]);

        let (started, mut starts) = mpsc::unbounded_channel();
        ahead.observe(
            &handle,
            Arc::from("object"),
            21,
            21,
            100,
            pending_fetch(started, cancelled),
        );
        assert_eq!(starts.recv().await, Some(22));
        assert_eq!(starts.recv().await, Some(23));

        drop(ahead);
        let mut pages = [cancels.recv().await, cancels.recv().await];
        pages.sort();
        assert_eq!(pages, [Some(22), Some(23)]);
    }

    #[tokio::test]
    async fn disabled_and_final_page_do_not_fetch() {
        let handle = TokioRuntime::current();
        let fetch = |_| -> BoxFuture<'static, ()> { unreachable!("no page follows") };
        ReadAhead::new(0, 2).observe(&handle, Arc::from("disabled"), 0, 0, 100, fetch);
        ReadAhead::new(4, 2).observe(&handle, Arc::from("end"), 9, 9, 10, fetch);
    }
}
