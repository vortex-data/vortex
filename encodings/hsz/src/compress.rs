// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::stage::BlockSummary;
use crate::stage::HSZ_BLOCK_SIZE;
use crate::stage::Hsz;

/// Configuration for [`Hsz::compress`].
#[derive(Clone, Copy, Debug)]
pub struct HszConfig {
    /// Absolute reconstruction error bound. Non-outlier elements decode to
    /// within `eps` of the original. Must be strictly positive and finite.
    pub eps: f64,
}

impl Default for HszConfig {
    /// Defaults: an `eps` of `1e-3`, a reasonable starting point for sensor
    /// and simulation data.
    fn default() -> Self {
        Self { eps: 1e-3 }
    }
}

/// Maximum residual width supported by the FastLanes packer for `u32`. We
/// cap at 31 because the FastLanes runtime dispatch table covers `0..32`
/// (the 32-bit identity is not in the table). Values that would need more
/// than 31 bits are routed to the outlier stage.
const MAX_BIT_WIDTH: u32 = 31;

impl Hsz {
    /// Compress an `f64` slice into a homomorphic encoding.
    ///
    /// Returns an [`Hsz`] whose [`Hsz::decompress`] is accurate to within
    /// `config.eps` for every non-outlier position.
    ///
    /// # Errors
    ///
    /// Returns an error if `config.eps` is non-positive or non-finite, or if
    /// the input exceeds `u32::MAX` elements.
    pub fn compress(values: &[f64], config: HszConfig) -> VortexResult<Self> {
        if !config.eps.is_finite() || config.eps <= 0.0 {
            vortex_bail!(
                "HszConfig::eps must be positive and finite, got {}",
                config.eps
            );
        }
        if values.len() > u32::MAX as usize {
            vortex_bail!(
                "Hsz currently supports at most {} elements, got {}",
                u32::MAX,
                values.len()
            );
        }

        let eps = config.eps;
        let n_blocks = values.len().div_ceil(HSZ_BLOCK_SIZE);

        let mut blocks: Vec<BlockSummary> = Vec::with_capacity(n_blocks);
        let mut block_starts: Vec<u32> = Vec::with_capacity(n_blocks + 1);
        block_starts.push(0);
        let mut bit_widths: Vec<u8> = Vec::with_capacity(n_blocks);
        let mut packed_offsets: Vec<u32> = Vec::with_capacity(n_blocks + 1);
        packed_offsets.push(0);
        let mut packed: BufferMut<u32> = BufferMut::with_capacity(0);
        let mut outlier_indices: Vec<u64> = Vec::new();
        let mut outlier_values: Vec<f64> = Vec::new();
        // Reusable per-block scratch space, always exactly 1024 u32 long.
        let mut residual_scratch = vec![0u32; HSZ_BLOCK_SIZE];

        for block_idx in 0..n_blocks {
            let start = block_idx * HSZ_BLOCK_SIZE;
            let end = (start + HSZ_BLOCK_SIZE).min(values.len());
            let block = &values[start..end];

            let mut summary = BlockSummary::empty();
            for &v in block {
                if v.is_finite() {
                    summary.observe(v);
                }
            }
            if summary.count == 0 {
                summary.min = 0.0;
                summary.max = 0.0;
            }

            let predictor = summary.min;
            let mut max_residual: u32 = 0;
            for (i, &v) in block.iter().enumerate() {
                let global_idx = (start + i) as u64;
                let q = quantise(v, predictor, eps);
                match q {
                    Some(r) => {
                        residual_scratch[i] = r;
                        if r > max_residual {
                            max_residual = r;
                        }
                    }
                    None => {
                        residual_scratch[i] = 0;
                        outlier_indices.push(global_idx);
                        outlier_values.push(v);
                    }
                }
            }
            // Zero the unused tail of the scratch buffer so the pack is
            // deterministic for partial blocks.
            for slot in &mut residual_scratch[block.len()..] {
                *slot = 0;
            }

            let bit_width = bit_width_for(max_residual);
            let block_packed_words = HSZ_BLOCK_SIZE * bit_width as usize / 32;
            let packed_start = packed.len();
            packed.push_n(0u32, block_packed_words);
            if bit_width > 0 {
                // SAFETY: the input slice is exactly HSZ_BLOCK_SIZE elements,
                // the output slice is exactly HSZ_BLOCK_SIZE * bit_width / 32
                // u32 words, and bit_width is in the supported 1..=31 range.
                unsafe {
                    <u32 as BitPacking>::unchecked_pack(
                        bit_width as usize,
                        &residual_scratch,
                        &mut packed.as_mut_slice()[packed_start..packed_start + block_packed_words],
                    );
                }
            }

            blocks.push(summary);
            block_starts.push(u32::try_from(start + block.len())?);
            bit_widths.push(bit_width);
            packed_offsets.push(u32::try_from(packed.len())?);
        }

        Ok(Hsz {
            eps,
            len: values.len(),
            blocks,
            block_starts,
            bit_widths,
            packed_offsets,
            packed: packed.freeze(),
            outlier_indices,
            outlier_values,
        })
    }
}

/// Quantise a single value against the block's predictor. Returns `None` if
/// the value cannot be represented within `eps` by a residual that fits the
/// packer (either non-finite, out of u31 range, or round-to-nearest
/// overshoots the error bound).
fn quantise(value: f64, predictor: f64, eps: f64) -> Option<u32> {
    if !value.is_finite() {
        return None;
    }
    let quantum = ((value - predictor) / eps).round();
    if quantum < 0.0 || quantum > (1u64 << MAX_BIT_WIDTH) as f64 - 1.0 {
        return None;
    }
    let r = quantum as u32;
    let reconstructed = predictor + (r as f64) * eps;
    if (reconstructed - value).abs() > eps {
        return None;
    }
    Some(r)
}

/// Smallest FastLanes-supported bit width that can hold `max_value`.
fn bit_width_for(max_value: u32) -> u8 {
    if max_value == 0 {
        0
    } else {
        (32 - max_value.leading_zeros()) as u8
    }
}
