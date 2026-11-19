// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A depth guard for the drop of deep trees.
//!
//! Dropping a tree node by node recurses once per level, which a tree deeper than the thread's
//! stack allows cannot afford. A shallow tree drops recursively, which is the cheapest way, and
//! past [`MAX_DROP_DEPTH`] nested drops a node switches to an iterative drop of its subtree.

use std::cell::Cell;

/// Nested recursive drops before a node switches to iterative cleanup.
const MAX_DROP_DEPTH: u32 = 32;

thread_local! {
    static DROP_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Holds a level of the thread's drop depth until it is dropped.
pub(crate) struct DropDepthGuard;

impl DropDepthGuard {
    /// Enter a level of recursive drop, or `None` when the thread is already [`MAX_DROP_DEPTH`]
    /// levels deep and the caller must drop iteratively.
    pub(crate) fn enter() -> Option<Self> {
        DROP_DEPTH.with(|depth| {
            let current = depth.get();
            (current < MAX_DROP_DEPTH).then(|| {
                depth.set(current + 1);
                Self
            })
        })
    }
}

impl Drop for DropDepthGuard {
    fn drop(&mut self) {
        DROP_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}
