// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tiny helpers shared between [`crate::scan`] and [`crate::handrolled`].

use std::time::Duration;

/// Median of a list of [`Duration`]s. Empty lists return `Duration::ZERO`.
pub fn median(runs: &[Duration]) -> Duration {
    if runs.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = runs.to_vec();
    sorted.sort();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        let total_nanos = sorted[mid - 1].as_nanos() + sorted[mid].as_nanos();
        Duration::from_nanos(u64::try_from(total_nanos / 2).unwrap_or(u64::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(median(&[]), Duration::ZERO);
    }

    #[test]
    fn odd_picks_middle() {
        let runs = [
            Duration::from_millis(1),
            Duration::from_millis(3),
            Duration::from_millis(2),
        ];
        assert_eq!(median(&runs), Duration::from_millis(2));
    }

    #[test]
    fn even_averages_middle_two() {
        let runs = [Duration::from_millis(2), Duration::from_millis(4)];
        assert_eq!(median(&runs), Duration::from_millis(3));
    }
}
