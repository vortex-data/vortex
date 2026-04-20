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
    #[expect(clippy::unwrap_in_result)]
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
