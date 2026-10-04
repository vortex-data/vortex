// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frame-of-reference bit packing for the per-block side data (block lengths and seeds).
//!
//! A packed sequence is an 8-byte little-endian `base`, a 1-byte bit `width`, then each value's
//! offset from `base` (wrapping) in `width` bits, least significant bit first.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::decode::read_bits;

const HEADER: usize = 9;

/// Pack `values` at the narrowest width that holds their offsets from the smallest.
pub(crate) fn pack(values: &[u64]) -> Vec<u8> {
    let base = values.iter().copied().min().unwrap_or(0);
    let width = values
        .iter()
        .map(|&v| 64 - (v - base).leading_zeros())
        .max()
        .unwrap_or(0);
    let mut out = Vec::with_capacity(packed_len(values.len(), width));
    out.extend_from_slice(&base.to_le_bytes());
    // A width is at most 64.
    #[allow(clippy::cast_possible_truncation)]
    out.push(width as u8);
    let (mut acc, mut filled) = (0u128, 0u32);
    for &v in values {
        acc |= u128::from(v - base) << filled;
        filled += width;
        while filled >= 8 {
            // Truncation takes the low byte.
            #[allow(clippy::cast_possible_truncation)]
            out.push(acc as u8);
            acc >>= 8;
            filled -= 8;
        }
    }
    if filled > 0 {
        #[allow(clippy::cast_possible_truncation)]
        out.push(acc as u8);
    }
    out
}

fn packed_len(n: usize, width: u32) -> usize {
    HEADER + (n * width as usize).div_ceil(8)
}

/// Check that `bytes` holds a packed sequence of exactly `n` values.
pub(crate) fn check(bytes: &[u8], n: usize) -> VortexResult<()> {
    vortex_ensure!(bytes.len() >= HEADER, "packed sequence lacks its header");
    let width = u32::from(bytes[8]);
    vortex_ensure!(width <= 64, "packed width {width} exceeds 64 bits");
    vortex_ensure!(
        bytes.len() == packed_len(n, width),
        "packed sequence of {n} values at {width} bits has {} bytes",
        bytes.len()
    );
    Ok(())
}

/// Unpack a sequence that passed [`check`].
pub(crate) fn unpack(bytes: &[u8], n: usize) -> Vec<u64> {
    let mut base = [0u8; 8];
    base.copy_from_slice(&bytes[..8]);
    let base = u64::from_le_bytes(base);
    let width = u32::from(bytes[8]);
    let packed = &bytes[HEADER..];
    (0..n)
        .map(|i| base.wrapping_add(read_bits(packed, i * width as usize, width)))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::*;

    #[rstest]
    #[case(vec![])]
    #[case(vec![7])]
    #[case(vec![5, 5, 5])]
    #[case(vec![300, 301, 1000, 299])]
    #[case(vec![0, u64::MAX, 1 << 63])]
    #[case((0..1000).map(|i| i * 7919 % 4093).collect())]
    fn roundtrip(#[case] values: Vec<u64>) -> VortexResult<()> {
        let bytes = pack(&values);
        check(&bytes, values.len())?;
        assert_eq!(unpack(&bytes, values.len()), values);
        if bytes[8] > 0 {
            assert!(check(&bytes, values.len() + 8).is_err());
        }
        Ok(())
    }
}
