// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::single_range_in_vec_init)]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;

use super::initial_ranges;

#[test]
fn initial_ranges_leave_a_disjoint_bulk_suffix() {
    let remaining = AtomicUsize::new(2);
    let initial = initial_ranges(100..30_000, &remaining);
    assert_eq!(initial, [100..8292, 8292..16_484]);
    assert_eq!(remaining.load(Ordering::Relaxed), 0);
    assert!(initial_ranges(0..30_000, &remaining).is_empty());
}

#[test]
fn short_files_leave_budget_for_the_next_file() {
    let remaining = AtomicUsize::new(2);
    assert!(initial_ranges(0..0, &remaining).is_empty());
    assert_eq!(initial_ranges(0..4000, &remaining), [0..4000]);
    assert_eq!(remaining.load(Ordering::Relaxed), 1);
    assert_eq!(initial_ranges(0..10_000, &remaining), [0..8192]);
    assert_eq!(remaining.load(Ordering::Relaxed), 0);
}

#[test]
fn initial_ranges_at_the_end_of_the_row_domain_do_not_overflow() {
    assert_eq!(
        initial_ranges(u64::MAX - 4..u64::MAX, &AtomicUsize::new(2)),
        [u64::MAX - 4..u64::MAX]
    );
}

#[test]
fn concurrent_files_share_one_budget() {
    let remaining = AtomicUsize::new(2);
    let claimed = AtomicUsize::new(0);
    thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                claimed.fetch_add(initial_ranges(0..4000, &remaining).len(), Ordering::Relaxed);
            });
        }
    });
    assert_eq!(remaining.load(Ordering::Relaxed), 0);
    assert_eq!(claimed.load(Ordering::Relaxed), 2);
}
