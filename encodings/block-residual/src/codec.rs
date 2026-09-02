// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use fastlanes::FoR as FastLanesFoR;
use vortex_error::VortexResult;

const CHUNK_LEN: usize = 1024;
const HIGH_PADDING: usize = 15;
const PATCH_DECODE_PENALTY_BITS: usize = 16;
const SERIALIZED_BLOCK_METADATA_BYTES: usize = 12;

/// Block-local residual codec for ordered unsigned latents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlockResidualCodec {
    len: usize,
    blocks: Vec<BlockResidualBlock>,
}

/// Serialized children for the one-reference block residual codec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlockResidualParts {
    pub(crate) len: usize,
    pub(crate) bases: Vec<u64>,
    pub(crate) residual_widths: Vec<u8>,
    pub(crate) high_widths: Vec<u8>,
    pub(crate) residual_starts: Vec<u32>,
    pub(crate) patch_starts: Vec<u32>,
    pub(crate) high_starts: Vec<u32>,
    pub(crate) residual_words: Vec<u64>,
    pub(crate) patch_positions: Vec<u16>,
    pub(crate) patch_highs: Vec<u8>,
}

pub(crate) struct BlockResidualCodecEstimate {
    pub encoded_nbytes: usize,
    pub patch_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BlockResidualBlock {
    base: u64,
    residual_width: u8,
    high_width: u8,
    residuals: Vec<u64>,
    patch_positions: Vec<u16>,
    patch_highs: Vec<u8>,
}

impl BlockResidualCodec {
    pub(crate) fn encode_with_word_width(values: &[u64], word_width: u8) -> VortexResult<Self> {
        vortex_error::vortex_ensure!(
            matches!(word_width, 8 | 16 | 32 | 64),
            "block residual word width is invalid"
        );
        let blocks = values
            .chunks(CHUNK_LEN)
            .map(|block| encode_block(block, word_width))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Self {
            len: values.len(),
            blocks,
        })
    }

    pub(crate) fn estimate_transformed<T: Copy>(
        values: &[T],
        transform: impl Fn(T) -> u64 + Copy,
    ) -> BlockResidualCodecEstimate {
        let mut encoded_nbytes = 3 * size_of::<u32>();
        let mut total_patch_count = 0;
        for values in values.chunks(CHUNK_LEN) {
            let plan = estimate_block(values, transform);
            let residual_nbytes = CHUNK_LEN * usize::from(plan.residual_width) / 8;
            let patch_high_nbytes = if plan.patch_count == 0 {
                0
            } else {
                (plan.patch_count * usize::from(plan.high_width)).div_ceil(8) + HIGH_PADDING
            };
            encoded_nbytes += size_of::<u64>()
                + 2 * size_of::<u8>()
                + 3 * size_of::<u32>()
                + residual_nbytes
                + plan.patch_count * size_of::<u16>()
                + patch_high_nbytes;
            total_patch_count += plan.patch_count;
        }
        BlockResidualCodecEstimate {
            encoded_nbytes,
            patch_count: total_patch_count,
        }
    }

    pub(crate) fn into_parts(self) -> VortexResult<BlockResidualParts> {
        let mut parts = BlockResidualParts {
            len: self.len,
            bases: Vec::with_capacity(self.blocks.len()),
            residual_widths: Vec::with_capacity(self.blocks.len()),
            high_widths: Vec::with_capacity(self.blocks.len()),
            residual_starts: Vec::with_capacity(self.blocks.len() + 1),
            patch_starts: Vec::with_capacity(self.blocks.len() + 1),
            high_starts: Vec::with_capacity(self.blocks.len() + 1),
            residual_words: Vec::new(),
            patch_positions: Vec::new(),
            patch_highs: Vec::new(),
        };
        parts.residual_starts.push(0);
        parts.patch_starts.push(0);
        parts.high_starts.push(0);
        for block in self.blocks {
            parts.bases.push(block.base);
            parts.residual_widths.push(block.residual_width);
            parts.high_widths.push(block.high_width);
            parts.residual_words.extend(block.residuals);
            parts.patch_positions.extend(block.patch_positions);
            parts.patch_highs.extend(block.patch_highs);
            parts
                .residual_starts
                .push(u32::try_from(parts.residual_words.len())?);
            parts
                .patch_starts
                .push(u32::try_from(parts.patch_positions.len())?);
            parts
                .high_starts
                .push(u32::try_from(parts.patch_highs.len())?);
        }
        Ok(parts)
    }
}
fn encode_block(values: &[u64], word_width: u8) -> VortexResult<BlockResidualBlock> {
    let base = values.iter().copied().min().unwrap_or(0);
    let mut residuals = Vec::with_capacity(CHUNK_LEN);
    let mut width_counts = [0usize; 65];
    let mut maximum_width = 0u8;
    for &value in values {
        let residual = value - base;
        let width = bit_width(residual);
        residuals.push(residual);
        width_counts[usize::from(width)] += 1;
        maximum_width = maximum_width.max(width);
    }
    residuals.resize(CHUNK_LEN, 0);

    let width_plan = choose_width(&width_counts, maximum_width, values.len());

    materialize_block(
        values,
        BlockPlan {
            base,
            residual_width: width_plan.residual_width,
            high_width: width_plan.high_width,
            residuals,
            patch_count: width_plan.patch_count,
        },
        word_width,
    )
}

struct BlockWidthPlan {
    residual_width: u8,
    high_width: u8,
    patch_count: usize,
}

fn estimate_block<T: Copy>(values: &[T], transform: impl Fn(T) -> u64) -> BlockWidthPlan {
    let base = values.iter().copied().map(&transform).min().unwrap_or(0);
    let mut width_counts = [0usize; 65];
    let mut maximum_width = 0u8;
    for &value in values {
        let width = bit_width(transform(value) - base);
        width_counts[usize::from(width)] += 1;
        maximum_width = maximum_width.max(width);
    }
    choose_width(&width_counts, maximum_width, values.len())
}

fn choose_width(
    width_counts: &[usize; 65],
    maximum_width: u8,
    value_count: usize,
) -> BlockWidthPlan {
    let mut patch_count = value_count;
    let mut best = (usize::MAX, maximum_width, 0u8, 0usize);
    for residual_width in 0..=maximum_width {
        patch_count -= width_counts[usize::from(residual_width)];
        let high_width = if patch_count == 0 {
            0
        } else {
            maximum_width - residual_width
        };
        let cost_bits = usize::from(residual_width) * CHUNK_LEN
            + patch_count * (u16::BITS as usize + usize::from(high_width))
            + patch_count * PATCH_DECODE_PENALTY_BITS
            + u64::BITS as usize
            + SERIALIZED_BLOCK_METADATA_BYTES * 8
            + usize::from(patch_count > 0) * HIGH_PADDING * 8;
        if cost_bits < best.0 {
            best = (cost_bits, residual_width, high_width, patch_count);
        }
    }
    BlockWidthPlan {
        residual_width: best.1,
        high_width: best.2,
        patch_count: best.3,
    }
}

struct BlockPlan {
    base: u64,
    residual_width: u8,
    high_width: u8,
    residuals: Vec<u64>,
    patch_count: usize,
}

fn materialize_block(
    values: &[u64],
    plan: BlockPlan,
    word_width: u8,
) -> VortexResult<BlockResidualBlock> {
    let residual_mask = low_mask(plan.residual_width);
    let low_residuals = plan
        .residuals
        .iter()
        .map(|&residual| residual & residual_mask)
        .collect::<Vec<_>>();
    let residuals = fast_pack(&low_residuals, plan.residual_width, word_width);
    let mut patch_positions = Vec::with_capacity(plan.patch_count);
    let mut patch_highs = BitWriter::with_capacity(plan.patch_count * 8);
    if plan.high_width > 0 {
        for (position, &residual) in plan.residuals[..values.len()].iter().enumerate() {
            let high = residual >> plan.residual_width;
            if high != 0 {
                patch_positions.push(u16::try_from(position)?);
                patch_highs.write(high, plan.high_width);
            }
        }
    }
    let patch_highs = if patch_positions.is_empty() {
        Vec::new()
    } else {
        let mut encoded = patch_highs.finish();
        encoded.extend_from_slice(&[0; HIGH_PADDING]);
        encoded
    };

    Ok(BlockResidualBlock {
        base: plan.base,
        residual_width: plan.residual_width,
        high_width: plan.high_width,
        residuals,
        patch_positions,
        patch_highs,
    })
}

fn fast_pack(values: &[u64], width: u8, word_width: u8) -> Vec<u64> {
    if width == 0 {
        return Vec::new();
    }
    match word_width {
        8 => fast_pack_native::<u8>(values, width),
        16 => fast_pack_native::<u16>(values, width),
        32 => fast_pack_native::<u32>(values, width),
        64 => fast_pack_native::<u64>(values, width),
        _ => unreachable!("validated block residual word width"),
    }
}

fn fast_pack_native<T: ResidualWord>(values: &[u64], width: u8) -> Vec<u64> {
    let mut packed_words = vec![0u64; CHUNK_LEN * usize::from(width) / u64::BITS as usize];
    let unpacked = values.iter().copied().map(T::from_u64).collect::<Vec<_>>();
    let packed_native = packed_words_as_native_mut::<T>(&mut packed_words);
    // SAFETY: Both slices have the exact lengths required for one FastLanes chunk.
    unsafe { T::unchecked_pack(usize::from(width), &unpacked, packed_native) };
    packed_words
}

pub(crate) trait ResidualWord: BitPacking + FastLanesFoR + Copy + Default {
    const BITS: u8;

    fn from_u64(value: u64) -> Self;

    fn to_u64(self) -> u64;

    fn wrapping_add(self, other: Self) -> Self;

    fn apply_high(&mut self, high: u64, shift: u8);

    unsafe fn unpack_add(bit_width: usize, packed: &[Self], base: Self, output: &mut [Self]);
}

macro_rules! impl_residual_word {
    ($T:ty, $bits:literal) => {
        impl ResidualWord for $T {
            const BITS: u8 = $bits;

            #[allow(clippy::cast_possible_truncation)]
            fn from_u64(value: u64) -> Self {
                value as $T
            }

            fn to_u64(self) -> u64 {
                u64::from(self)
            }

            fn wrapping_add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }

            #[allow(clippy::cast_possible_truncation)]
            fn apply_high(&mut self, high: u64, shift: u8) {
                *self |= (high as $T) << shift;
            }

            unsafe fn unpack_add(
                bit_width: usize,
                packed: &[Self],
                base: Self,
                output: &mut [Self],
            ) {
                // SAFETY: The caller provides one complete FastLanes input and output chunk.
                unsafe { FastLanesFoR::unchecked_unfor_pack(bit_width, packed, base, output) };
            }
        }
    };
}

impl_residual_word!(u8, 8);
impl_residual_word!(u16, 16);
impl_residual_word!(u32, 32);
impl_residual_word!(u64, 64);

pub(crate) fn packed_words_as_native<T: ResidualWord>(words: &[u64]) -> &[T] {
    // SAFETY: Unsigned integer types permit every bit pattern. A `u64` slice has sufficient
    // alignment, and packed FastLanes payloads always contain a whole number of bytes.
    let (prefix, native, suffix) = unsafe { words.align_to::<T>() };
    debug_assert!(prefix.is_empty() && suffix.is_empty());
    native
}

fn packed_words_as_native_mut<T: ResidualWord>(words: &mut [u64]) -> &mut [T] {
    // SAFETY: Unsigned integer types permit every bit pattern. A `u64` slice has sufficient
    // alignment, and packed FastLanes payloads always contain a whole number of bytes.
    let (prefix, native, suffix) = unsafe { words.align_to_mut::<T>() };
    debug_assert!(prefix.is_empty() && suffix.is_empty());
    native
}

fn bit_width(value: u64) -> u8 {
    u8::try_from(u64::BITS - value.leading_zeros()).unwrap_or(64)
}

fn low_mask(bits: u8) -> u64 {
    match bits {
        0 => 0,
        64 => u64::MAX,
        _ => (1_u64 << bits) - 1,
    }
}

pub(crate) unsafe fn read_wide_bits(bytes: &[u8], bit_position: usize, width: u8) -> u64 {
    let byte_position = bit_position / 8;
    let bits_past_byte = bit_position % 8;
    // SAFETY: The caller provides fifteen readable padding bytes.
    let first = unsafe { read_u64_unaligned(bytes.as_ptr().add(byte_position)) };
    if width <= 57 {
        (first >> bits_past_byte) & low_mask(width)
    } else {
        // SAFETY: The caller provides fifteen readable padding bytes.
        let second = unsafe { read_u64_unaligned(bytes.as_ptr().add(byte_position + 7)) };
        let processed = 56 - bits_past_byte;
        ((first >> bits_past_byte) | (second << processed)) & low_mask(width)
    }
}

unsafe fn read_u64_unaligned(pointer: *const u8) -> u64 {
    // SAFETY: The caller provides eight readable bytes at the pointer.
    u64::from_le(unsafe { pointer.cast::<u64>().read_unaligned() })
}

struct BitWriter {
    bytes: Vec<u8>,
    pending: u64,
    pending_bits: u8,
}

impl BitWriter {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
            pending: 0,
            pending_bits: 0,
        }
    }

    fn write(&mut self, value: u64, width: u8) {
        if width == 0 {
            return;
        }
        let available = 64 - self.pending_bits;
        self.pending |= value << self.pending_bits;
        if width < available {
            self.pending_bits += width;
            return;
        }
        self.bytes.extend_from_slice(&self.pending.to_le_bytes());
        let remaining = width - available;
        self.pending = if remaining == 0 {
            0
        } else {
            value >> available
        };
        self.pending_bits = remaining;
    }

    fn finish(mut self) -> Vec<u8> {
        if self.pending_bits > 0 {
            let byte_count = usize::from(self.pending_bits).div_ceil(8);
            self.bytes
                .extend_from_slice(&self.pending.to_le_bytes()[..byte_count]);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::BlockResidualCodec;

    #[test]
    fn patch_penalty_prefers_dense_residuals() -> VortexResult<()> {
        let mut values = vec![0_u64; 1_024];
        values[..307].fill(4_095);
        let parts = BlockResidualCodec::encode_with_word_width(&values, 64)?.into_parts()?;

        assert_eq!(parts.residual_widths, [12]);
        assert!(parts.patch_positions.is_empty());
        Ok(())
    }
}
