// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use parking_lot::RwLock;
use sketches_ddsketch::DDSketch;

/// Stores an arbitrary number of data points, giving approximated information about its distribution.
/// The current implementation uses an implementation of the [DDSketch] type but that might change in the future.
///
/// [DDSketch]: https://arxiv.org/pdf/1908.10693
#[derive(Default, Clone)]
pub struct Histogram(Arc<RwLock<DDSketch>>);

impl std::fmt::Debug for Histogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Histogram").finish_non_exhaustive()
    }
}

impl Histogram {
    pub(crate) fn new() -> Self {
        Self(Default::default())
    }

    /// Adds a sample to the histogram
    pub fn update(&self, value: f64) {
        self.0.write().add(value);
    }

    /// Returns the estimated quantile value, which must be in the [0.0, 1.0] range, will panic otherwise.
    /// Returns `None` if the histogram is empty.
    #[expect(clippy::expect_used)]
    pub fn quantile(&self, quantile: f64) -> Option<f64> {
        assert!(
            (0.0..=1.0).contains(&quantile),
            "quantile must be between 0.0 and 1.0"
        );

        self.0
            .read()
            .quantile(quantile)
            .expect("quantile range checked")
    }

    /// Returns the sum of all values stored in the histogram
    pub fn total(&self) -> f64 {
        self.0.read().sum().unwrap_or_default()
    }

    /// Returns the number of values recorded.
    pub fn count(&self) -> usize {
        self.0.read().count()
    }

    /// Returns true if the histogram contains 0 samples.
    pub fn is_empty(&self) -> bool {
        self.0.read().count() == 0
    }
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;

    use super::Histogram;

    #[test]
    fn histogram_empty_state() {
        let histogram = Histogram::new();
        assert!(histogram.is_empty());
        assert_eq!(histogram.count(), 0);
        assert_eq!(histogram.total(), 0.0);
        assert_eq!(histogram.quantile(0.0), None);
        assert_eq!(histogram.quantile(0.5), None);
        assert_eq!(histogram.quantile(1.0), None);
    }

    #[test]
    fn histogram_updates_quantiles_and_total() {
        let histogram = Histogram::new();
        for value in 1..=100 {
            histogram.update(f64::from(value));
        }

        assert!(!histogram.is_empty());
        assert_eq!(histogram.count(), 100);
        assert!((histogram.total() - 5050.0).abs() < 1e-6);

        let min = histogram.quantile(0.0).unwrap();
        let median = histogram.quantile(0.5).unwrap();
        let max = histogram.quantile(1.0).unwrap();
        assert!(min > 0.0 && min <= median && median <= max);
    }

    #[test]
    fn histogram_zero_and_extreme_values() {
        let histogram = Histogram::new();
        histogram.update(0.0);
        assert!(!histogram.is_empty());
        assert_eq!(histogram.count(), 1);
        assert_eq!(histogram.total(), 0.0);
        assert_eq!(histogram.quantile(0.5), Some(0.0));

        histogram.update(1e-6);
        histogram.update(1e9);
        assert_eq!(histogram.count(), 3);
        assert!(histogram.quantile(1.0).unwrap() > 1e8);
    }

    #[test]
    fn histogram_clone_shares_state() {
        let histogram = Histogram::new();
        let cloned = histogram.clone();

        histogram.update(10.0);
        cloned.update(20.0);

        assert_eq!(histogram.count(), 2);
        assert_eq!(cloned.count(), 2);
        assert!((histogram.total() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn histogram_out_of_range_quantiles_panic() {
        for invalid in [-0.01, 1.01, f64::NAN] {
            let err = catch_unwind(|| Histogram::new().quantile(invalid)).unwrap_err();
            let msg = err
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| err.downcast_ref::<String>().map(String::as_str))
                .unwrap_or_default();
            assert!(
                msg.contains("quantile must be between 0.0 and 1.0"),
                "unexpected panic for {invalid}: {msg}"
            );
        }
    }
}
