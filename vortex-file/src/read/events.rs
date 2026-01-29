// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use futures::StreamExt;
use futures::channel::mpsc::TrySendError;
use futures::channel::mpsc::UnboundedReceiver;
use futures::channel::mpsc::UnboundedSender;
use futures::channel::mpsc::unbounded;
use futures::stream::FusedStream;
use vortex_error::VortexExpect;

use crate::segments::ReadEvent;

pub struct EventsChannel;

pub struct EventsReceiver {
    inner: Option<UnboundedReceiver<ReadEvent>>,
    num_senders: Arc<AtomicUsize>,
    is_done: bool,
}

impl EventsReceiver {
    fn new(inner: UnboundedReceiver<ReadEvent>, num_senders: Arc<AtomicUsize>) -> Self {
        Self {
            inner: Some(inner),
            num_senders,
            is_done: false,
        }
    }

    fn terminate(&mut self) {
        self.is_done = true;
        self.inner.take();
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
        if self.is_done || self.inner.as_ref().is_some_and(|rx| rx.is_terminated()) {
            self.terminate();
            return Poll::Ready(None);
        }

        if self.num_senders.load(Ordering::SeqCst) == 0 {
            self.terminate();

            return Poll::Ready(None);
        }

        self.inner
            .as_mut()
            .vortex_expect("Must exist here")
            .poll_next_unpin(cx)
    }
}

impl FusedStream for EventsReceiver {
    fn is_terminated(&self) -> bool {
        self.is_done || self.inner.as_ref().is_some_and(|rx| rx.is_terminated())
    }
}

#[cfg(test)]
mod tests {

    use futures::future::FusedFuture;

    use super::*;

    #[tokio::test]
    async fn test_cancellation_no_senders() -> anyhow::Result<()> {
        let (tx, mut rx) = EventsChannel::unbounded();
        tx.unbounded_send(ReadEvent::Polled(1))?;
        tx.unbounded_send(ReadEvent::Polled(2))?;
        tx.unbounded_send(ReadEvent::Polled(3))?;
        let tx2 = tx.clone();
        tx2.unbounded_send(ReadEvent::Polled(4))?;

        assert!(rx.next().await.is_some());
        assert!(rx.next().await.is_some());

        drop(tx);
        assert!(rx.next().await.is_some());
        drop(tx2);

        // We technically still have one event, but we stop anyway.
        assert!(rx.next().await.is_none());
        assert!(rx.next().is_terminated());
        assert!(rx.next().await.is_none());

        Ok(())
    }
}
