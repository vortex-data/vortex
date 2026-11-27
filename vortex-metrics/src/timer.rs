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

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;
    use std::time::Duration;

    use super::Timer;

    #[test]
    fn timer_empty_state() {
        let timer = Timer::new();
        assert!(timer.is_empty());
        assert_eq!(timer.count(), 0);
        assert_eq!(timer.total(), Duration::ZERO);
        assert_eq!(timer.quantile(0.0), None);
        assert_eq!(timer.quantile(0.5), None);
        assert_eq!(timer.quantile(1.0), None);
    }

    #[test]
    fn timer_update_quantiles_and_total() {
        let timer = Timer::new();
        for ms in 1..=100 {
            timer.update(Duration::from_millis(ms));
        }

        assert!(!timer.is_empty());
        assert_eq!(timer.count(), 100);
        assert!(timer.total().abs_diff(Duration::from_millis(5050)) < Duration::from_micros(100));

        let min = timer.quantile(0.0).unwrap();
        let median = timer.quantile(0.5).unwrap();
        let max = timer.quantile(1.0).unwrap();
        assert!(
            (Duration::from_micros(500)..=Duration::from_micros(1500)).contains(&min),
            "unexpected min: {min:?}"
        );
        assert!(
            (Duration::from_millis(45)..=Duration::from_millis(55)).contains(&median),
            "unexpected median: {median:?}"
        );
        assert!(
            (Duration::from_millis(95)..=Duration::from_millis(105)).contains(&max),
            "unexpected max: {max:?}"
        );
    }

    #[test]
    fn timer_zero_and_nanosecond_durations() {
        let timer = Timer::new();
        timer.update(Duration::ZERO);
        assert_eq!(timer.count(), 1);
        assert_eq!(timer.total(), Duration::ZERO);
        assert_eq!(timer.quantile(0.5), Some(Duration::ZERO));

        timer.update(Duration::from_nanos(500));
        assert_eq!(timer.count(), 2);
        assert_eq!(timer.total(), Duration::from_nanos(500));
    }

    #[test]
    fn timer_raii_guard_records_on_drop() {
        let timer = Timer::new();
        let cloned = timer.clone();

        let guard = timer.time();
        assert!(timer.is_empty());
        assert_eq!(timer.count(), 0);
        drop(guard);

        assert!(!timer.is_empty());
        assert_eq!(timer.count(), 1);
        assert_eq!(cloned.count(), 1);
        assert!(timer.quantile(0.5).is_some());

        {
            let _second = cloned.time();
            assert_eq!(timer.count(), 1);
        }
        assert_eq!(timer.count(), 2);
    }

    #[test]
    fn timer_out_of_range_quantiles_panic() {
        for invalid in [-0.1, 1.1, f64::NAN] {
            let err = catch_unwind(|| Timer::new().quantile(invalid)).unwrap_err();
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
