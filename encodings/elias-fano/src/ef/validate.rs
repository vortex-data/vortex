// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checking a stored layout against the universe it claims.
//!
//! Every stored number is recomputed from `(span, n)` alone and a layout that disagrees is refused,
//! which is what lets the sample tables carry no seam and every sampled search skip its bounds
//! checks.
//!
//! The upper array's contents are *not* checked: that costs a pass over the whole bit array, and a
//! reader raises rather than underflowing on a position it cannot invert.

use super::LOG_SAMPLING0;
use super::LOG_SAMPLING1;
use super::Malformed;
use super::Table;
use super::lower_width;
use super::num_samples0;
use super::num_samples1;
use super::num_zeros;
use super::read_sample;
use super::upper_len;

/// Check that a stored layout describes the universe it claims.
///
/// `samples` is the shared table, zero-samples first; the seam between the two is derived here
/// rather than stored, so a buffer too short to reach it fails as a count mismatch.
///
/// An empty sequence is accepted unconditionally, having no geometry to check.
pub fn validate_layout(
    span: u64,
    n: usize,
    stored_lower_width: u8,
    stored_upper_len: u64,
    samples: &[u8],
) -> Result<(), Malformed> {
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

    // Both tables sample from rank 1 upward, so their sizes follow from the layout and a reader
    // never has to bounds-check a lookup.
    let expected_samples0 = num_samples0(span, expected_width);
    debug_assert_eq!(
        expected_samples0,
        (num_zeros(expected_upper_len, n) - 1) >> LOG_SAMPLING0,
        "the two derivations of the zero-sample count must agree"
    );
    let expected_samples1 = num_samples1(n);
    let expected = expected_samples0 + expected_samples1;
    let found = (samples.len() / size_of::<u64>()) as u64;
    if found != expected {
        return Err(Malformed::SampleCount { expected, found });
    }

    // The zero table comes first. The count check above already proves the buffer reaches the seam.
    let seam = (expected_samples0 as usize) * size_of::<u64>();
    let (samples0, samples1) = samples.split_at(seam);

    // A sample is fed straight to `select_range` as a window start, which asserts rather than
    // raises past the end, so every one is checked — there are only `n / 256 + zeros / 512`. Each
    // is pinned above by the upper array's length and below by the rank it stands for, which gives
    // the strict increase a sampled search relies on.
    for (table, name, log_sampling, floor) in [
        (samples0, Table::Zero, LOG_SAMPLING0, 0),
        (samples1, Table::One, LOG_SAMPLING1, 1),
    ] {
        let mut previous = None;
        for index in 0..table.len() / size_of::<u64>() {
            let sample = read_sample(table, index);
            let minimum = (((index + 1) as u64) << log_sampling) + floor;
            if !(minimum..expected_upper_len).contains(&sample) {
                return Err(Malformed::SampleOutOfRange {
                    table: name,
                    index,
                    sample,
                    min: minimum,
                    max: expected_upper_len,
                });
            }
            if previous.is_some_and(|previous| previous >= sample) {
                return Err(Malformed::SamplesNotIncreasing { table: name, index });
            }
            previous = Some(sample);
        }
    }

    Ok(())
}
