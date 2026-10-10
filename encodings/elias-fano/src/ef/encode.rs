// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building the layout from a non-decreasing sequence of elements.

use super::Error;
use super::UpperBuilder;
use super::lower_mask;
use super::lower_width;
use super::position_of;
use super::upper_len;

/// The buffers an encoded sequence occupies.
///
/// Every field is a plain `Vec`, which a caller with its own buffer type can adopt rather than copy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded {
    /// The upper bit array, LSB-first, padded to whole bytes.
    pub upper: Vec<u8>,
    /// Zero-samples followed by one-samples, sharing one table. The seam is not recorded: a reader
    /// recovers it from the universe with [`num_samples0`](super::num_samples0).
    pub samples: Vec<u64>,
    /// The low [`lower_width`](Self::lower_width) bits of every element, in element order. Empty of
    /// meaning at width zero, where every entry is `0`.
    pub lower: Vec<u64>,
    /// Low bits per element.
    pub lower_width: u8,
    /// Length of the upper array in bits, which is not `upper.len() * 8`.
    pub upper_len: u64,
}

/// Encode a non-decreasing sequence of elements spanning `0..=span`.
///
/// An *element* is a value with the sequence's reference already subtracted, so `span` is the last
/// element.
///
/// A caller holding values must check monotonicity **in the value domain** before subtracting: an
/// element is a modular difference, so an unsorted sequence can still yield non-decreasing elements
/// after wrapping, which nothing here can detect afterwards.
///
/// # Panics
///
/// Panics in debug builds if `elements` is empty, or if it is not non-decreasing, or if its last
/// element is not `span`. A release build produces a layout no reader will accept.
pub fn encode(elements: impl ExactSizeIterator<Item = u64>, span: u64) -> Result<Encoded, Error> {
    let n = elements.len();
    debug_assert!(n > 0, "the empty sequence has no layout to build");

    let lower_width = lower_width(span, n);
    let upper_len = upper_len(span, n, lower_width)?;
    let mask = lower_mask(lower_width);

    // `upper_len` has already refused a length `usize` cannot address.
    let bits = usize::try_from(upper_len).map_err(|_| Error::UpperLenTooLarge { upper_len })?;
    let mut upper = UpperBuilder::new(bits);
    let mut lower = Vec::with_capacity(n);
    let mut previous = 0u64;

    for (index, element) in elements.enumerate() {
        debug_assert!(element >= previous, "elements must be non-decreasing");
        debug_assert!(element <= span, "element {element} exceeds the span {span}");
        previous = element;

        let rank = index as u64;
        upper.push(rank, position_of(element, rank, lower_width));
        lower.push(element & mask);
    }

    debug_assert_eq!(lower.len(), n, "ExactSizeIterator yielded the wrong count");
    debug_assert_eq!(previous, span, "the last element is the span");

    let (upper, samples) = upper.finish(n as u64, upper_len);
    Ok(Encoded {
        upper,
        samples,
        lower,
        lower_width,
        upper_len,
    })
}
