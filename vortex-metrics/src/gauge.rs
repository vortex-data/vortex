// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

/// A gauge metric that can increase or decrease, representing a single value at the point of sampling.
#[derive(Clone)]
pub struct Gauge(Arc<AtomicU64>);

impl std::fmt::Debug for Gauge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Gauge").field(&self.value()).finish()
    }
}

impl Gauge {
    pub(crate) fn new() -> Self {
        Self(Default::default())
    }

    /// Increment the gauge by value.
    pub fn increment(&self, value: f64) {
        loop {
            if self
                .0
                .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |current| {
                    let input = f64::from_bits(current);
                    Some((input + value).to_bits())
                })
                .is_ok()
            {
                break;
            }
        }
    }

    /// Decrement the gauge by value.
    pub fn decrement(&self, value: f64) {
        loop {
            if self
                .0
                .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |current| {
                    let input = f64::from_bits(current);
                    Some((input - value).to_bits())
                })
                .is_ok()
            {
                break;
            }
        }
    }

    /// Sets the gauge to a specific value.
    pub fn set(&self, value: f64) {
        // We use `swap` with `AcqRel` ordering to make sure we get
        // consistent ordering across operations.
        _ = self.0.swap(value.to_bits(), Ordering::AcqRel);
    }

    /// Returns the current value of the gauge.
    pub fn value(&self) -> f64 {
        let value = self.0.load(Ordering::Acquire);
        f64::from_bits(value)
    }
}
