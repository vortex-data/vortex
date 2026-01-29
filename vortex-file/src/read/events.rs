use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::ready,
    task::{Context, Poll},
};

use futures::{
    Stream, StreamExt,
    channel::mpsc::{TrySendError, UnboundedReceiver, UnboundedSender, unbounded},
    stream::FusedStream,
};

use crate::segments::ReadEvent;

pub struct EventsChannel;

pub struct EventsReceiver {
    inner: UnboundedReceiver<ReadEvent>,
    num_senders: Arc<AtomicUsize>,
    is_done: bool,
}

impl EventsReceiver {
    fn new(inner: UnboundedReceiver<ReadEvent>, num_senders: Arc<AtomicUsize>) -> Self {
        Self {
            inner,
            num_senders,
            is_done: false,
        }
    }
}

pub struct EventsSender {
    inner: UnboundedSender<ReadEvent>,
    num_senders: Arc<AtomicUsize>,
}

impl Clone for EventsSender {
    fn clone(&self) -> Self {
        self.num_senders.fetch_add(1, Ordering::SeqCst);
        Self {
            inner: self.inner.clone(),
            num_senders: self.num_senders.clone(),
        }
    }
}

impl Drop for EventsSender {
    fn drop(&mut self) {
        self.num_senders.fetch_sub(1, Ordering::SeqCst);
    }
}

impl EventsSender {
    fn new(inner: UnboundedSender<ReadEvent>, num_senders: Arc<AtomicUsize>) -> Self {
        Self { inner, num_senders }
    }

    pub fn unbounded_send(&self, read_event: ReadEvent) -> Result<(), TrySendError<ReadEvent>> {
        self.inner.unbounded_send(read_event)
    }
}

impl EventsChannel {
    pub fn unbounded() -> (EventsSender, EventsReceiver) {
        let (tx, rx) = unbounded();
        let num_senders = Arc::new(AtomicUsize::new(1));

        (
            EventsSender::new(tx, num_senders.clone()),
            EventsReceiver::new(rx, num_senders),
        )
    }
}

impl Stream for EventsReceiver {
    type Item = ReadEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_done || self.inner.is_terminated() {
            return Poll::Ready(None);
        }

        if self.num_senders.load(Ordering::SeqCst) == 0 {
            if std::env::var("PANIC_EVENTS").is_ok() {
                panic!("Oops");
            }

            self.is_done = true;
            return Poll::Ready(None);
        }

        return self.inner.poll_next_unpin(cx);
    }
}

impl FusedStream for EventsReceiver {
    fn is_terminated(&self) -> bool {
        self.is_done || self.inner.is_terminated()
    }
}
