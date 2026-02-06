// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

/// A commutative value that can only be increased, and starts at 0 on initialization.
#[derive(Clone, Debug)]
pub struct Counter(Arc<AtomicU64>);

impl Counter {
    pub(crate) fn new() -> Self {
        Self(Default::default())
    }

    /// Adds `value` to the counter
    pub fn add(&self, value: u64) {
        self.0.fetch_add(value, Ordering::Release);
    }

    /// Returns the latest value stored in the counter
    pub fn value(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
}
