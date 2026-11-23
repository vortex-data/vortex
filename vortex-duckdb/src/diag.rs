// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diagnostic work counters for the DuckDB scan, printed once per scan. Not for merging.

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

pub(crate) static FILES: AtomicU64 = AtomicU64::new(0);
pub(crate) static SKIPPED: AtomicU64 = AtomicU64::new(0);
pub(crate) static PARTS: AtomicU64 = AtomicU64::new(0);
pub(crate) static JOINS: AtomicU64 = AtomicU64::new(0);
pub(crate) static WASTED: AtomicU64 = AtomicU64::new(0);
pub(crate) static NSI: AtomicU64 = AtomicU64::new(0);
pub(crate) static ARRAYS: AtomicU64 = AtomicU64::new(0);
pub(crate) static BLOCK_ON: AtomicU64 = AtomicU64::new(0);
pub(crate) static PENDED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static YIELDED: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn inc(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn joined() {
    inc(&JOINS);
    YIELDED.with(|y| y.set(false));
}

pub(crate) fn yielded() {
    inc(&ARRAYS);
    YIELDED.with(|y| y.set(true));
}

pub(crate) fn split_done() {
    if !YIELDED.with(Cell::get) {
        inc(&WASTED);
    }
}

/// Counts a `block_on` entry, and whether its future's first poll pended.
pub(crate) struct Counted<F> {
    fut: F,
    polled: bool,
}

pub(crate) fn counted<F: Future>(fut: F) -> Counted<F> {
    inc(&BLOCK_ON);
    Counted { fut, polled: false }
}

impl<F: Future + Unpin> Future for Counted<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let first = !self.polled;
        self.polled = true;
        let poll = Pin::new(&mut self.fut).poll(cx);
        if first && poll.is_pending() {
            inc(&PENDED);
        }
        poll
    }
}

/// Prints and resets the counters.
pub(crate) fn report() {
    let take = |c: &AtomicU64| c.swap(0, Ordering::Relaxed);
    eprintln!(
        "VXDIAG files={} skipped={} parts={} joins={} wasted_joins={} next_split_item={} arrays={} block_on={} block_on_pended={}",
        take(&FILES),
        take(&SKIPPED),
        take(&PARTS) + vortex::scan::DIAG_PARTS.swap(0, Ordering::Relaxed),
        take(&JOINS),
        take(&WASTED),
        take(&NSI),
        take(&ARRAYS),
        take(&BLOCK_ON),
        take(&PENDED),
    );
}
