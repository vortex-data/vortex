// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use parking_lot::Mutex;

/// A byte budget that admits chunks into the write pipeline in FIFO order.
///
/// Chunks are admitted strictly in the order they ask, and the pipeline behind the budget
/// releases them in the same order, so a waiting chunk can never be starved by chunks admitted
/// after it. A chunk larger than the whole budget is admitted once nothing else is in flight,
/// so a single oversized chunk cannot wait forever.
pub(crate) struct InFlightBudget {
    max_bytes: u64,
    state: Mutex<State>,
}

struct State {
    used_bytes: u64,
    next_ticket: u64,
    waiters: VecDeque<Waiter>,
}

struct Waiter {
    ticket: u64,
    waker: Option<Waker>,
}

impl State {
    fn admits(&self, max_bytes: u64, bytes: u64) -> bool {
        self.used_bytes == 0 || self.used_bytes.saturating_add(bytes) <= max_bytes
    }

    fn wake_front(&mut self) {
        if let Some(waker) = self.waiters.front_mut().and_then(|w| w.waker.take()) {
            waker.wake();
        }
    }
}

impl InFlightBudget {
    pub(crate) fn new(max_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            max_bytes,
            state: Mutex::new(State {
                used_bytes: 0,
                next_ticket: 0,
                waiters: VecDeque::new(),
            }),
        })
    }

    /// Returns the byte budget.
    pub(crate) fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Returns the number of bytes currently admitted and not yet released.
    pub(crate) fn used_bytes(&self) -> u64 {
        self.state.lock().used_bytes
    }

    /// Waits for `bytes` of budget, returning a permit that returns them when dropped.
    pub(crate) fn acquire(self: &Arc<Self>, bytes: u64) -> Acquire {
        Acquire {
            budget: Arc::clone(self),
            bytes,
            ticket: None,
        }
    }
}

/// A future resolving to an [`InFlightPermit`] once the budget admits the request.
pub(crate) struct Acquire {
    budget: Arc<InFlightBudget>,
    bytes: u64,
    ticket: Option<u64>,
}

impl Future for Acquire {
    type Output = InFlightPermit;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        let max_bytes = this.budget.max_bytes;
        let bytes = this.bytes;
        let mut state = this.budget.state.lock();

        let is_front = match this.ticket {
            None => state.waiters.is_empty(),
            Some(ticket) => state.waiters.front().is_some_and(|w| w.ticket == ticket),
        };
        if is_front && state.admits(max_bytes, bytes) {
            if this.ticket.take().is_some() {
                state.waiters.pop_front();
            }
            state.used_bytes += bytes;
            // The next waiter may also fit, e.g. after a release large enough for two.
            state.wake_front();
            return Poll::Ready(InFlightPermit {
                budget: Arc::clone(&this.budget),
                bytes,
            });
        }

        match this.ticket {
            None => {
                let ticket = state.next_ticket;
                state.next_ticket += 1;
                state.waiters.push_back(Waiter {
                    ticket,
                    waker: Some(cx.waker().clone()),
                });
                this.ticket = Some(ticket);
            }
            Some(ticket) => {
                if let Some(waiter) = state.waiters.iter_mut().find(|w| w.ticket == ticket) {
                    waiter.waker = Some(cx.waker().clone());
                }
            }
        }
        Poll::Pending
    }
}

impl Drop for Acquire {
    fn drop(&mut self) {
        let Some(ticket) = self.ticket else {
            return;
        };
        let mut state = self.budget.state.lock();
        let was_front = state.waiters.front().is_some_and(|w| w.ticket == ticket);
        state.waiters.retain(|w| w.ticket != ticket);
        if was_front {
            state.wake_front();
        }
    }
}

/// Bytes admitted into the pipeline, returned to the budget when dropped.
pub(crate) struct InFlightPermit {
    budget: Arc<InFlightBudget>,
    bytes: u64,
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        let mut state = self.budget.state.lock();
        state.used_bytes -= self.bytes;
        state.wake_front();
    }
}

#[cfg(test)]
mod tests {
    use futures::FutureExt;

    use super::InFlightBudget;

    #[test]
    fn admits_until_the_budget_is_spent() {
        let budget = InFlightBudget::new(10);
        let first = budget.acquire(6).now_or_never().expect("fits");
        assert_eq!(budget.used_bytes(), 6);

        let mut second = Box::pin(budget.acquire(6));
        assert!(second.as_mut().now_or_never().is_none());

        drop(first);
        let second = second
            .as_mut()
            .now_or_never()
            .expect("admitted after release");
        assert_eq!(budget.used_bytes(), 6);
        drop(second);
    }

    #[test]
    fn admits_in_fifo_order() {
        let budget = InFlightBudget::new(10);
        let first = budget.acquire(8).now_or_never().expect("fits");

        let mut second = Box::pin(budget.acquire(8));
        assert!(second.as_mut().now_or_never().is_none());
        // Fits right now, but must wait behind the earlier request.
        let mut third = Box::pin(budget.acquire(1));
        assert!(third.as_mut().now_or_never().is_none());

        drop(first);
        assert!(third.as_mut().now_or_never().is_none());
        let second = second.as_mut().now_or_never().expect("admitted first");
        let third = third.as_mut().now_or_never().expect("admitted second");
        assert_eq!(budget.used_bytes(), 9);
        drop(second);
        drop(third);
    }

    #[test]
    fn oversized_request_is_admitted_when_idle() {
        let budget = InFlightBudget::new(4);
        let permit = budget
            .acquire(100)
            .now_or_never()
            .expect("idle budget admits");
        assert_eq!(budget.used_bytes(), 100);

        let mut next = Box::pin(budget.acquire(1));
        assert!(next.as_mut().now_or_never().is_none());
        drop(permit);
        let next = next.as_mut().now_or_never().expect("admitted once idle");
        assert_eq!(budget.used_bytes(), 1);
        drop(next);
    }

    #[test]
    fn dropped_waiter_unblocks_the_queue() {
        let budget = InFlightBudget::new(10);
        let first = budget.acquire(8).now_or_never().expect("fits");
        let mut second = Box::pin(budget.acquire(8));
        assert!(second.as_mut().now_or_never().is_none());
        let mut third = Box::pin(budget.acquire(1));
        assert!(third.as_mut().now_or_never().is_none());

        drop(second);
        let third = third
            .as_mut()
            .now_or_never()
            .expect("admitted once unblocked");
        assert_eq!(budget.used_bytes(), 9);
        drop(third);
        drop(first);
    }
}
