// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-width residuals from a linear trend, for the per-block index.
//!
//! Block byte offsets, tANS stream lengths and delta initial values are close to linear in the
//! block index for most data, so storing `v[i] - (base + i * slope)` in a fixed bit width keeps
//! O(1) access while costing next to nothing for regular data.

use crate::bits::BitWriter;
use crate::bits::PADDING;
use crate::bits::read_bits_wide;
use crate::latent::offset_bits;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Packed {
    len: usize,
    base: u64,
    slope: u64,
    width: u8,
    /// Residual bits, followed by [`PADDING`] zero bytes.
    bits: Vec<u8>,
}

impl Packed {
    pub fn new(values: &[u64]) -> Self {
        let Some((&first, &last)) = values.first().zip(values.last()) else {
            return Self {
                bits: vec![0; PADDING],
                ..Default::default()
            };
        };
        let n = values.len() as i128;
        let slope = if n > 1 {
            (i128::from(last) - i128::from(first)).div_euclid(n - 1)
        } else {
            0
        };
        let fit = |slope: i128| -> Option<(i128, u8)> {
            let detrended = values
                .iter()
                .enumerate()
                .map(|(i, &v)| i128::from(v) - slope * i as i128);
            let (lo, hi) = detrended.fold((i128::MAX, i128::MIN), |(lo, hi), t| {
                (lo.min(t), hi.max(t))
            });
            let range = u64::try_from(hi - lo).ok()?;
            Some((lo, offset_bits(range)))
        };
        // A trend can only help; plain frame-of-reference always fits in 64 bits.
        let (slope, (base, width)) = match (fit(slope), fit(0)) {
            (Some(trend), Some(flat)) if trend.1 < flat.1 => (slope, trend),
            (_, Some(flat)) => (0, flat),
            (Some(trend), None) => (slope, trend),
            (None, None) => (0, (0, 64)),
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (base, slope) = (base as u64, slope as u64);

        let mut writer = BitWriter::default();
        for (i, &v) in values.iter().enumerate() {
            let residual = v
                .wrapping_sub(base)
                .wrapping_sub(slope.wrapping_mul(i as u64));
            writer.write(residual, u32::from(width));
        }
        let mut bits = writer.finish();
        bits.extend_from_slice(&[0; PADDING]);
        Self {
            len: values.len(),
            base,
            slope,
            width,
            bits,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn width(&self) -> u8 {
        self.width
    }

    #[inline]
    pub fn get(&self, i: usize) -> u64 {
        assert!(i < self.len, "packed index {i} out of bounds for {}", self.len);
        // SAFETY: `i < len`, so the read starts inside the residual bits, which are followed by
        // `PADDING` bytes.
        let residual = unsafe { read_bits_wide(&self.bits, i * usize::from(self.width), self.width) };
        self.base
            .wrapping_add(self.slope.wrapping_mul(i as u64))
            .wrapping_add(residual)
    }

    pub(crate) fn base(&self) -> u64 {
        self.base
    }

    pub(crate) fn slope(&self) -> u64 {
        self.slope
    }

    /// The residual bits, without padding.
    pub(crate) fn residual_bytes(&self) -> &[u8] {
        &self.bits[..self.bits.len() - PADDING]
    }

    /// Bytes of residuals for `len` values of `width` bits.
    pub(crate) fn residual_len(len: usize, width: u8) -> usize {
        (len * usize::from(width)).div_ceil(8)
    }

    /// Rebuilds from serialized parts, copying the residuals into a padded buffer.
    pub(crate) fn from_parts(
        len: usize,
        base: u64,
        slope: u64,
        width: u8,
        residuals: &[u8],
    ) -> vortex_error::VortexResult<Self> {
        vortex_error::vortex_ensure!(width <= 64, "packed width {width} exceeds 64");
        vortex_error::vortex_ensure!(
            residuals.len() == Self::residual_len(len, width),
            "expected {} residual bytes, got {}",
            Self::residual_len(len, width),
            residuals.len()
        );
        let mut bits = Vec::with_capacity(residuals.len() + PADDING);
        bits.extend_from_slice(residuals);
        bits.extend_from_slice(&[0; PADDING]);
        Ok(Self {
            len,
            base,
            slope,
            width,
            bits,
        })
    }

    /// Serialized size: base, slope (zigzag) and width as varints, and the residual bits. The
    /// length is implied by the owning stream.
    pub fn nbytes(&self) -> usize {
        let zigzag = |v: u64| (v << 1) ^ ((v.cast_signed() >> 63).cast_unsigned());
        varint_len(self.base) + varint_len(zigzag(self.slope)) + 1 + self.bits.len() - PADDING
    }
}

/// Bytes in the LEB128 encoding of `v`, as protobuf stores integers.
pub(crate) fn varint_len(v: u64) -> usize {
    (u64::BITS - v.leading_zeros()).max(1).div_ceil(7) as usize
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(vec![])]
    #[case(vec![7])]
    #[case((0..1000).map(|i| 1_000_000 + i * 60_000).collect())]
    #[case((0..1000).map(|i| i * 137 + (i * 7919) % 13).collect())]
    #[case(vec![u64::MAX, 0, u64::MAX / 2, 5])]
    #[case((0..100).map(|i| u64::MAX - i * 3).collect())]
    fn roundtrip(#[case] values: Vec<u64>) {
        let packed = Packed::new(&values);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(packed.get(i), v);
        }
    }

    #[test]
    fn linear_costs_nothing() {
        let values: Vec<u64> = (0..5000).map(|i| 42 + i * 1234).collect();
        let packed = Packed::new(&values);
        assert_eq!(packed.width(), 0);
        assert_eq!(packed.nbytes(), 1 + 2 + 1);
    }
}
