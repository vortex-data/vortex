// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_buffer::CpuKernel;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::AllOr;
use crate::Mask;
use crate::MaskValues;

impl Mask {
    /// Pack `refined` into the dense rank domain selected by `self`.
    ///
    /// `refined` must cover the same rows and may only select rows selected by `self`. The output
    /// has length `self.true_count()` and contains, in order, the `refined` bit corresponding to
    /// each selected row in `self`.
    pub fn compress_by_mask(&self, refined: &Self) -> VortexResult<Self> {
        if self.len() != refined.len() {
            return Err(vortex_err!(
                "mask compression inputs have different lengths: {} and {}",
                self.len(),
                refined.len()
            ));
        }

        let output_len = self.true_count();
        if refined.all_false() {
            return Ok(Self::new_false(output_len));
        }
        if self.all_true() {
            return Ok(refined.clone());
        }
        if self.all_false() || refined.all_true() {
            return Err(vortex_err!(
                "refined mask selects rows outside the compression mask"
            ));
        }
        if self.true_count() == refined.true_count() {
            if !refined.is_subset_of(self) {
                return Err(vortex_err!(
                    "refined mask selects rows outside the compression mask"
                ));
            }
            return Ok(Self::new_true(output_len));
        }

        let (AllOr::Some(selector), AllOr::Some(source)) =
            (self.bit_buffer(), refined.bit_buffer())
        else {
            unreachable!("constant masks were handled by fast paths")
        };
        let buffer = compress_bitbuffers(source, selector, output_len)
            .ok_or_else(|| vortex_err!("refined mask selects rows outside the compression mask"))?;
        Ok(mask_from_buffer(buffer, refined.true_count()))
    }
}

type CompressKernel = unsafe fn(&BitBuffer, &BitBuffer, usize) -> Option<BitBuffer>;

fn compress_bitbuffers(
    source: &BitBuffer,
    selector: &BitBuffer,
    output_len: usize,
) -> Option<BitBuffer> {
    static KERNEL: CpuKernel<CompressKernel> = CpuKernel::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if bmi2_kernel_available() {
                return compress_bmi2;
            }
        }
        compress_fallback
    });
    // SAFETY: the selector returns either a safe fallback or a kernel whose required CPU feature
    // was checked before selection.
    unsafe { KERNEL.get()(source, selector, output_len) }
}

#[cfg(target_arch = "x86_64")]
fn bmi2_kernel_available() -> bool {
    std::arch::is_x86_feature_detected!("bmi2") && std::arch::is_x86_feature_detected!("popcnt")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,popcnt")]
unsafe fn compress_bmi2(
    source: &BitBuffer,
    selector: &BitBuffer,
    output_len: usize,
) -> Option<BitBuffer> {
    use std::arch::x86_64::_pext_u64;

    compress_inner(source, selector, output_len, |bits, mask| {
        _pext_u64(bits, mask)
    })
}

fn compress_fallback(
    source: &BitBuffer,
    selector: &BitBuffer,
    output_len: usize,
) -> Option<BitBuffer> {
    compress_inner(source, selector, output_len, pext_fallback)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the low word is emitted before the accumulator is shifted"
)]
fn compress_inner(
    source: &BitBuffer,
    selector: &BitBuffer,
    output_len: usize,
    extract: impl Fn(u64, u64) -> u64,
) -> Option<BitBuffer> {
    debug_assert_eq!(source.len(), selector.len());
    let source_chunks = source.chunks();
    let selector_chunks = selector.chunks();
    let mut output = BufferMut::<u64>::with_capacity(output_len.div_ceil(64));
    let mut accumulator = 0u128;
    let mut accumulator_bits = 0u32;

    let mut append = |source_word: u64, selector_word: u64| -> Option<()> {
        if source_word & !selector_word != 0 {
            return None;
        }
        let selected = selector_word.count_ones();
        if selected == 0 {
            return Some(());
        }
        let extracted = if selector_word == u64::MAX {
            source_word
        } else {
            extract(source_word, selector_word)
        };
        accumulator |= (extracted as u128) << accumulator_bits;
        accumulator_bits += selected;
        if accumulator_bits >= 64 {
            output.push(accumulator as u64);
            accumulator >>= 64;
            accumulator_bits -= 64;
        }
        Some(())
    };

    for (source_word, selector_word) in source_chunks.iter().zip(selector_chunks.iter()) {
        append(source_word, selector_word)?;
    }
    append(
        source_chunks.remainder_bits(),
        selector_chunks.remainder_bits(),
    )?;
    if accumulator_bits > 0 {
        output.push(accumulator as u64);
    }

    let mut bytes = output.into_byte_buffer();
    bytes.truncate(output_len.div_ceil(8));
    Some(BitBuffer::new(bytes.freeze(), output_len))
}

#[inline(always)]
fn pext_fallback(source: u64, mask: u64) -> u64 {
    let source = source.to_le_bytes();
    let mask = mask.to_le_bytes();
    let mut result = 0u64;
    let mut offset = 0u32;
    for idx in 0..8 {
        let selector = mask[idx];
        if selector != 0 {
            let extracted = BYTE_PEXT_LUT[(usize::from(selector) << 8) | usize::from(source[idx])];
            result |= u64::from(extracted) << offset;
            offset += selector.count_ones();
        }
    }
    result
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "both table construction indices are bounded below 256"
)]
static BYTE_PEXT_LUT: &[u8; 256 * 256] = &{
    let mut table = [0u8; 256 * 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut source = 0usize;
        while source < 256 {
            let mut selector = mask as u8;
            let mut input_bit = 0u8;
            let mut output_bit = 0u8;
            let mut extracted = 0u8;
            while selector != 0 {
                if selector & 1 != 0 {
                    if source as u8 & (1 << input_bit) != 0 {
                        extracted |= 1 << output_bit;
                    }
                    output_bit += 1;
                }
                selector >>= 1;
                input_bit += 1;
            }
            table[(mask << 8) | source] = extracted;
            source += 1;
        }
        mask += 1;
    }
    table
};

fn mask_from_buffer(buffer: BitBuffer, true_count: usize) -> Mask {
    let len = buffer.len();
    if true_count == 0 {
        return Mask::new_false(len);
    }
    if true_count == len {
        return Mask::new_true(len);
    }
    Mask::Values(Arc::new(MaskValues {
        buffer,
        indices: Default::default(),
        slices: Default::default(),
        true_count,
        density: true_count as f64 / len as f64,
    }))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use crate::Mask;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn bmi2_dispatch_checks_every_enabled_target_feature() {
        if super::bmi2_kernel_available() {
            assert!(std::arch::is_x86_feature_detected!("bmi2"));
            assert!(std::arch::is_x86_feature_detected!("popcnt"));
        }
    }

    #[rstest]
    #[case([true, true, true, true, true], [true, false, true, false, true])]
    #[case([false, true, false, true, false], [false, true, false, false, false])]
    #[case([true, false, false, false, true], [false, false, false, false, true])]
    #[case([false, false, false, false, false], [false, false, false, false, false])]
    fn compresses_by_rank(
        #[case] selector: [bool; 5],
        #[case] refined: [bool; 5],
    ) -> VortexResult<()> {
        let expected = selector
            .iter()
            .zip(refined)
            .filter_map(|(selected, refined)| selected.then_some(refined));
        let actual = Mask::from_iter(selector).compress_by_mask(&Mask::from_iter(refined))?;
        assert_eq!(actual, Mask::from_iter(expected));
        Ok(())
    }

    #[test]
    fn compresses_misaligned_dense_and_sparse_views() -> VortexResult<()> {
        let selector_bits = BitBuffer::from_iter((0..143).map(|idx| idx % 3 == 0 || idx % 11 == 0));
        let refined_bits = BitBuffer::from_iter((0..143).map(|idx| idx % 11 == 0));
        let selector = Mask::from_buffer(selector_bits.slice(5..138));
        let refined = Mask::from_buffer(refined_bits.slice(5..138));
        let expected = selector
            .iter()
            .zip(refined.iter())
            .filter_map(|(selected, refined)| selected.then_some(refined));

        assert_eq!(
            selector.compress_by_mask(&refined)?,
            Mask::from_iter(expected)
        );
        Ok(())
    }

    #[test]
    fn rejects_bits_outside_selector() {
        let selector = Mask::from_iter([true, false, true]);
        let refined = Mask::from_iter([true, true, false]);
        assert!(selector.compress_by_mask(&refined).is_err());
    }

    #[test]
    fn matches_scalar_reference_across_lengths_densities_and_offsets() -> VortexResult<()> {
        for len in [0, 1, 7, 63, 64, 65, 257] {
            for offset in [0, 1, 5, 7] {
                for pattern in 0..4 {
                    let total = len + offset + 3;
                    let selected = |idx: usize| match pattern {
                        0 => true,
                        1 => idx.is_multiple_of(2),
                        2 => idx.is_multiple_of(17),
                        _ => !idx.is_multiple_of(10),
                    };
                    let kept = |idx: usize| {
                        selected(idx)
                            && match pattern {
                                0 => idx.is_multiple_of(3),
                                1 => idx.is_multiple_of(4),
                                2 => idx.is_multiple_of(51),
                                _ => idx.is_multiple_of(3),
                            }
                    };
                    let selector = Mask::from_buffer(
                        BitBuffer::from_iter((0..total).map(selected)).slice(offset..offset + len),
                    );
                    let refined = Mask::from_buffer(
                        BitBuffer::from_iter((0..total).map(kept)).slice(offset..offset + len),
                    );
                    let expected = selector
                        .iter()
                        .zip(refined.iter())
                        .filter_map(|(selected, refined)| selected.then_some(refined));

                    assert_eq!(
                        selector.compress_by_mask(&refined)?,
                        Mask::from_iter(expected),
                        "len={len}, offset={offset}, pattern={pattern}"
                    );
                }
            }
        }
        Ok(())
    }
}
