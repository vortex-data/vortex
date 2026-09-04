// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::time::Duration;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::display::IndentedFormatter;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;
use crate::display::profile::DecompressionProfile;
use crate::display::profile::NodeTiming;
use crate::display::profile::ProfileOptions;

/// Extractor that adds a `throughput:` detail line from a measured [`DecompressionProfile`].
///
/// The line reports the time to canonicalize the subtree, its share of the whole tree's time, the
/// rates that time implies, and either the node's self time or the amount of child work it fuses
/// into itself. A share above 100% means the child costs more on its own than the parent that
/// fuses it.
///
/// Nodes missing from the profile are left unannotated, so a profile may be rendered against a
/// subtree of the tree it was measured on.
pub struct ThroughputExtractor {
    profile: DecompressionProfile,
}

impl ThroughputExtractor {
    /// Annotate a tree with an already-measured profile.
    pub fn new(profile: DecompressionProfile) -> Self {
        Self { profile }
    }

    /// Measure `array` and annotate it with the result.
    pub fn measure(
        array: &ArrayRef,
        session: &VortexSession,
        options: ProfileOptions,
    ) -> VortexResult<Self> {
        Ok(Self::new(DecompressionProfile::measure(
            array, session, options,
        )?))
    }

    /// The profile backing this extractor.
    pub fn profile(&self) -> &DecompressionProfile {
        &self.profile
    }
}

impl TreeExtractor<ArrayRef, TreeContext> for ThroughputExtractor {
    fn write_details(
        &self,
        array: &ArrayRef,
        _ctx: &TreeContext,
        f: &mut IndentedFormatter<'_, '_>,
    ) -> fmt::Result {
        let Some(timing) = self.profile.get(array) else {
            return Ok(());
        };
        let (indent, f) = f.parts();
        writeln!(
            f,
            "{indent}throughput: {}",
            Timing(timing, self.profile.root_time())
        )
    }
}

struct Timing<'a>(&'a NodeTiming, Duration);

impl fmt::Display for Timing<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(timing, root) = *self;
        let percent = if root.is_zero() {
            0.0
        } else {
            100_f64 * timing.subtree.as_secs_f64() / root.as_secs_f64()
        };
        write!(
            f,
            "{} ({percent:.2}%) | in {} | out {} | {}",
            Elapsed(timing.subtree),
            Rate(timing.input_bytes_per_sec(), &["B", "kB", "MB", "GB"]),
            Rate(timing.output_bytes_per_sec(), &["B", "kB", "MB", "GB"]),
            Rate(timing.rows_per_sec(), &["row", "krow", "Mrow", "Grow"]),
        )?;
        match timing.fusion_saving() {
            Some(saving) => write!(f, " | fuses children (saves {})", Elapsed(saving)),
            None => write!(f, " | self {}", Elapsed(timing.self_time())),
        }
    }
}

/// A duration, rendered as `1.81ms`.
struct Elapsed(Duration);

impl fmt::Display for Elapsed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0.as_secs_f64();
        for (scale, unit) in [(1.0, "s"), (1e-3, "ms"), (1e-6, "µs")] {
            if secs >= scale {
                return write!(f, "{:.2}{unit}", secs / scale);
            }
        }
        write!(f, "{:.0}ns", secs * 1e9)
    }
}

/// A per-second rate, rendered in the largest unit that keeps it above one, e.g. `1.90 GB/s`.
struct Rate(f64, &'static [&'static str]);

impl fmt::Display for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(rate, units) = *self;
        if !rate.is_finite() {
            return write!(f, "n/a");
        }
        let mut scale = 1.0;
        let mut unit = units[0];
        for next in &units[1..] {
            if rate < scale * 1e3 {
                break;
            }
            scale *= 1e3;
            unit = next;
        }
        write!(f, "{:.2} {unit}/s", rate / scale)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(Duration::from_nanos(0), "0ns")]
    #[case(Duration::from_nanos(640), "640ns")]
    #[case(Duration::from_nanos(1_500), "1.50µs")]
    #[case(Duration::from_micros(1_810), "1.81ms")]
    #[case(Duration::from_millis(2_500), "2.50s")]
    fn elapsed_picks_a_unit(#[case] elapsed: Duration, #[case] expected: &str) {
        assert_eq!(Elapsed(elapsed).to_string(), expected);
    }

    #[rstest]
    #[case(0.0, "0.00 B/s")]
    #[case(999.0, "999.00 B/s")]
    #[case(1_000.0, "1.00 kB/s")]
    #[case(1.9e9, "1.90 GB/s")]
    // Rates beyond the largest unit keep that unit rather than wrapping around.
    #[case(2e12, "2000.00 GB/s")]
    #[case(f64::INFINITY, "n/a")]
    fn rate_picks_a_unit(#[case] rate: f64, #[case] expected: &str) {
        assert_eq!(Rate(rate, &["B", "kB", "MB", "GB"]).to_string(), expected);
    }
}
