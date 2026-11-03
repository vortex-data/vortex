// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Debug-only counters shared by all files and workers of one V2 table scan.
pub(crate) struct Diagnostics {
    pub(super) id: u64,
    started: Instant,
    first_array: AtomicBool,
    first_export: AtomicBool,
    arrays: AtomicU64,
    array_rows: AtomicU64,
    batches: AtomicU64,
    returned_rows: AtomicU64,
    compute_ns: AtomicU64,
    io_wait_ns: AtomicU64,
}

impl Diagnostics {
    pub(crate) fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            started: Instant::now(),
            first_array: AtomicBool::new(false),
            first_export: AtomicBool::new(false),
            arrays: AtomicU64::new(0),
            array_rows: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            returned_rows: AtomicU64::new(0),
            compute_ns: AtomicU64::new(0),
            io_wait_ns: AtomicU64::new(0),
        }
    }

    pub(super) fn array(&self, rows: usize) {
        if !self.first_array.swap(true, Ordering::Relaxed) {
            tracing::debug!(
                scan_id = self.id,
                elapsed_us = self.started.elapsed().as_micros(),
                rows,
                "v2 first native array"
            );
        }
        self.arrays.fetch_add(1, Ordering::Relaxed);
        self.array_rows.fetch_add(rows as u64, Ordering::Relaxed);
        tracing::debug!(
            scan_id = self.id,
            elapsed_us = self.started.elapsed().as_micros(),
            rows,
            "v2 native array returned"
        );
    }

    pub(crate) fn exported(&self, rows: u64) {
        if rows == 0 {
            return;
        }
        if !self.first_export.swap(true, Ordering::Relaxed) {
            tracing::debug!(
                scan_id = self.id,
                elapsed_us = self.started.elapsed().as_micros(),
                rows,
                "v2 first DuckDB batch"
            );
        }
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.returned_rows.fetch_add(rows, Ordering::Relaxed);
    }

    pub(super) fn compute(&self, ns: u128) {
        self.compute_ns
            .fetch_add(u64::try_from(ns).unwrap_or(u64::MAX), Ordering::Relaxed);
    }
    pub(super) fn io_wait(&self, ns: u128) {
        self.io_wait_ns
            .fetch_add(u64::try_from(ns).unwrap_or(u64::MAX), Ordering::Relaxed);
    }
}

impl Drop for Diagnostics {
    fn drop(&mut self) {
        tracing::debug!(
            scan_id = self.id,
            elapsed_us = self.started.elapsed().as_micros(),
            arrays = self.arrays.load(Ordering::Relaxed),
            array_rows = self.array_rows.load(Ordering::Relaxed),
            batches = self.batches.load(Ordering::Relaxed),
            returned_rows = self.returned_rows.load(Ordering::Relaxed),
            compute_ns = self.compute_ns.load(Ordering::Relaxed),
            io_wait_ns = self.io_wait_ns.load(Ordering::Relaxed),
            "v2 scan summary"
        );
    }
}

/// Records bound changes when Vortex reads DuckDB's shared dynamic filter.
pub(crate) struct FilterObserver {
    started: Instant,
    previous: parking_lot::Mutex<Option<Option<vortex::scalar::ScalarValue>>>,
    reads: AtomicU64,
    read_ns: AtomicU64,
    updates: AtomicU64,
    column: String,
}

impl FilterObserver {
    pub(crate) fn new(column: String) -> Self {
        Self {
            started: Instant::now(),
            previous: parking_lot::Mutex::new(None),
            reads: AtomicU64::new(0),
            read_ns: AtomicU64::new(0),
            updates: AtomicU64::new(0),
            column,
        }
    }

    pub(crate) fn observe(&self, value: Option<&vortex::scalar::ScalarValue>, read_ns: u128) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.read_ns.fetch_add(
            u64::try_from(read_ns).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        let mut previous = self.previous.lock();
        if previous.as_ref().map(Option::as_ref) != Some(value) {
            self.updates.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(elapsed_us = self.started.elapsed().as_micros(), column = self.column, bound = ?value, previous = ?previous.as_ref(), "v2 dynamic filter observed");
            *previous = Some(value.cloned());
        }
    }
}

impl Drop for FilterObserver {
    fn drop(&mut self) {
        tracing::debug!(
            column = self.column,
            reads = self.reads.load(Ordering::Relaxed),
            read_ns = self.read_ns.load(Ordering::Relaxed),
            observations = self.updates.load(Ordering::Relaxed),
            "v2 dynamic filter summary"
        );
    }
}
