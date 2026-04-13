// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use parking_lot::RwLock;
use sketches_ddsketch::DDSketch;

/// A specialized histogram for storing timed measurements. Like [`Histogram`], it uses [DDSketch] to store approximated values
/// but accepts and returns nano-scale durations.
///
/// [`Histogram`]: crate::Histogram
#[derive(Clone, Default)]
pub struct Timer(Arc<RwLock<DDSketch>>);

impl std::fmt::Debug for Timer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Timer").finish_non_exhaustive()
    }
}

impl Timer {
    pub(crate) fn new() -> Self {
        Self(Default::default())
    }

    /// Record a duration.
    pub fn update(&self, duration: Duration) {
        self.0.write().add(duration.as_secs_f64());
    }

    /// Returns the sum of all recorded durations.
    pub fn total(&self) -> Duration {
        self.0
            .read()
            .sum()
            .map(Duration::from_secs_f64)
            .unwrap_or_default()
    }

    /// Returns the estimated quantile value, which must be in the [0.0, 1.0] range, will panic otherwise.
    /// Returns `None` if the timer is empty.
    #[expect(clippy::expect_used)]
    #[expect(clippy::unwrap_in_result)]
    pub fn quantile(&self, quantile: f64) -> Option<Duration> {
        assert!(
            (0.0..=1.0).contains(&quantile),
            "quantile must be between 0.0 and 1.0"
        );

        self.0
            .read()
            .quantile(quantile)
            .expect("quantile range checked")
            .map(Duration::from_secs_f64)
    }

    /// Returns the number of values recorded.
    pub fn count(&self) -> usize {
        self.0.read().count()
    }

    /// Returns true if the timer contains 0 samples.
    pub fn is_empty(&self) -> bool {
        self.0.read().count() == 0
    }

    /// Returns a RAII guard that starts measuring time, recording time passed between it being created to being dropped.
    pub fn time(&self) -> TimeGuard<'_> {
        TimeGuard {
            source: self,
            start: Instant::now(),
        }
    }
}

/// RAII guard attached to a [`Timer`] instance, will record the time passed since its creation when dropped.
pub struct TimeGuard<'a> {
    source: &'a Timer,
    start: Instant,
}

impl Drop for TimeGuard<'_> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        self.source.update(elapsed);
    }
}
