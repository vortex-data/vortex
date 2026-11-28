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
                .try_update(Ordering::AcqRel, Ordering::Relaxed, |current| {
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
                .try_update(Ordering::AcqRel, Ordering::Relaxed, |current| {
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

#[cfg(test)]
mod tests {
    use std::thread;

    use super::Gauge;

    #[test]
    fn gauge_set_increment_decrement_and_negative_values() {
        let gauge = Gauge::new();
        assert_eq!(gauge.value(), 0.0);

        gauge.increment(2.5);
        assert_eq!(gauge.value(), 2.5);

        gauge.decrement(5.0);
        assert_eq!(gauge.value(), -2.5);

        gauge.increment(-1.5);
        assert_eq!(gauge.value(), -4.0);

        gauge.decrement(-2.0);
        assert_eq!(gauge.value(), -2.0);

        gauge.set(42.25);
        assert_eq!(gauge.value(), 42.25);

        gauge.set(-0.0);
        assert!(gauge.value().is_sign_negative());
        assert_eq!(gauge.value(), 0.0);
    }

    #[test]
    fn gauge_clone_shares_state() {
        let gauge = Gauge::new();
        let cloned = gauge.clone();

        gauge.set(10.5);
        assert_eq!(cloned.value(), 10.5);

        cloned.decrement(0.5);
        assert_eq!(gauge.value(), 10.0);
    }

    #[test]
    fn gauge_concurrent_updates() {
        let gauge = Gauge::new();
        thread::scope(|scope| {
            for _ in 0..4 {
                let g = gauge.clone();
                scope.spawn(move || {
                    for _ in 0..500 {
                        g.increment(2.0);
                    }
                });
            }
            for _ in 0..4 {
                let g = gauge.clone();
                scope.spawn(move || {
                    for _ in 0..500 {
                        g.decrement(1.0);
                    }
                });
            }
        });
        assert_eq!(gauge.value(), 4.0 * 500.0);
    }
}
