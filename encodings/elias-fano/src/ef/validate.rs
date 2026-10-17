// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checking a stored layout against the universe it claims.
//!
//! Every stored number is recomputed from `(span, n)` alone and a layout that disagrees is refused,
//! which is what lets every sampled search skip its bounds checks. A window of the sequence is
//! checked against the same geometry: its bytes and samples must lie inside the whole array's.
//!
//! The upper array's contents are *not* checked: that costs a pass over the whole bit array, and a
//! reader raises rather than underflowing on a position it cannot invert.

use super::LOG_SAMPLING0;
use super::LOG_SAMPLING1;
use super::Malformed;
use super::Table;
use super::Window;
use super::lower_width;
use super::num_samples0;
use super::num_samples1;
use super::num_zeros;
use super::upper_len;

/// Check that a stored window describes the universe it claims.
///
/// A window with `upper_base` 0 must hold the whole array: `upper_len` bits rounded up to whole
/// bytes and both tables complete. A later window must fit inside that array and hold the
/// one-samples its ranks need. Zero-sample coverage needs a select to check, so it is left to the
/// readers, which raise on a missing sample.
///
/// An empty sequence is accepted unconditionally, having no geometry to check.
pub fn validate_layout(
    span: u64,
    n: usize,
    stored_lower_width: u8,
    stored_upper_len: u64,
    window: &Window<'_>,
) -> Result<(), Malformed> {
    let (base, first_rank, len) = (window.upper_base, window.first_rank, window.len);
    if first_rank
        .checked_add(len as u64)
        .is_none_or(|end| end > n as u64)
    {
        return Err(Malformed::RankRange { first_rank, len, n });
    }
    if n == 0 {
        return Ok(());
    }

    let expected_width = lower_width(span, n);
    if stored_lower_width != expected_width {
        return Err(Malformed::LowerWidth {
            expected: expected_width,
            found: stored_lower_width,
        });
    }

    let expected_upper_len = upper_len(span, n, expected_width)?;
    if stored_upper_len != expected_upper_len {
        return Err(Malformed::UpperLen {
            expected: expected_upper_len,
            found: stored_upper_len,
        });
    }

    // The bits before a window are its first rank's ones and some of the zeros. This also makes a
    // window at bit 0 start at rank 0.
    let zeros = num_zeros(expected_upper_len, n);
    if first_rank > base || base - first_rank > zeros {
        return Err(Malformed::UpperBase {
            upper_base: base,
            first_rank,
        });
    }

    // Readers wrap the window in a `Bits`, which asserts that the bytes hold it.
    let expected_bytes = expected_upper_len.div_ceil(8);
    let bytes = window.upper.len() as u64;
    let held = bytes
        .saturating_mul(8)
        .saturating_sub(base % 8)
        .min(expected_upper_len - base);
    if base == 0 {
        if bytes != expected_bytes {
            return Err(Malformed::UpperBytes {
                expected: expected_bytes,
                found: bytes,
            });
        }
    } else if base / 8 + bytes > expected_bytes || held < len as u64 {
        return Err(Malformed::UpperWindow {
            upper_base: base,
            bytes,
            upper_len: expected_upper_len,
        });
    }

    let full0 = num_samples0(span, expected_width);
    debug_assert_eq!(
        full0,
        (zeros - 1) >> LOG_SAMPLING0,
        "the two derivations of the zero-sample count must agree"
    );
    let full1 = num_samples1(n);
    let needed1 = len
        .checked_sub(1)
        .map_or(0, |last| (first_rank + last as u64) >> LOG_SAMPLING1);

    // Every sample is checked, as there are only `n / 256 + zeros / 512`: each lies in the window,
    // no lower than the rank it stands for, and each table strictly increases, which lets
    // `slice_bounds` check only the ends of the samples it keeps.
    for (samples, table, log_sampling, floor, full, needed) in [
        (window.samples0(), Table::Zero, LOG_SAMPLING0, 0, full0, 0),
        (
            window.samples1(),
            Table::One,
            LOG_SAMPLING1,
            1,
            full1,
            needed1,
        ),
    ] {
        let first = samples.first();
        let count = samples.len() as u64;
        if base == 0 {
            if count != full {
                return Err(Malformed::SampleCount {
                    table,
                    expected: full,
                    found: count,
                });
            }
        } else if first + count > full || first + count < needed {
            return Err(Malformed::SampleWindow {
                table,
                end: first + count,
                min: needed.max(first),
                max: full,
            });
        }

        let mut previous = None;
        for local in 0..samples.len() {
            let index = first + local as u64;
            let sample = samples.get(local).unwrap_or(u64::MAX);
            let min = base.max(((index + 1) << log_sampling) + floor);
            let max = base + held;
            if !(min..max).contains(&sample) {
                return Err(Malformed::SampleOutOfRange {
                    table,
                    index,
                    sample,
                    min,
                    max,
                });
            }
            if previous.is_some_and(|previous| previous >= sample) {
                return Err(Malformed::SamplesNotIncreasing { table, index });
            }
            previous = Some(sample);
        }
    }

    Ok(())
}
