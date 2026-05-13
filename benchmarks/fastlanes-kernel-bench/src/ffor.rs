// SPDX-FileCopyrightText: Copyright the FastLanes Authors
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vendored verbatim from `fastlanes` 0.5.0 / src/ffor.rs.

use crate::{pack, seq_t, supported_bit_width, unpack, BitPacking, FastLanes};
use arrayref::{array_mut_ref, array_ref};
use paste::paste;

pub trait FoR: BitPacking {
    fn for_pack<const W: usize, const B: usize>(
        input: &[Self; 1024],
        reference: Self,
        output: &mut [Self; B],
    );

    fn unfor_pack<const W: usize, const B: usize>(
        input: &[Self; B],
        reference: Self,
        output: &mut [Self; 1024],
    );

    /// Fused unpack and wrapping add a `FoR` reference value.
    /// Unpacks 1024 elements from `W` bits each, where `W` is runtime-known instead of
    /// compile-time known.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output slice must be of exactly length 1024.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unfor_pack(
        width: usize,
        input: &[Self],
        reference: Self,
        output: &mut [Self],
    );
}

macro_rules! impl_for {
    ($T:ty) => {
        impl FoR for $T {
            fn for_pack<const W: usize, const B: usize>(
                input: &[Self; 1024],
                reference: Self,
                output: &mut [Self; B],
            ) {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                for lane in 0..Self::LANES {
                    pack!($T, W, output, lane, |$idx| {
                        input[$idx].wrapping_sub(reference)
                    });
                }
            }

            #[inline(never)]
            fn unfor_pack<const W: usize, const B: usize>(
                input: &[Self; B],
                reference: Self,
                output: &mut [Self; 1024],
            ) {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                for lane in 0..Self::LANES {
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        output[$idx] = $elem.wrapping_add(reference)
                    });
                }
            }

           unsafe fn unchecked_unfor_pack(width: usize, input: &[Self], reference: Self, output: &mut [Self]) {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::unfor_pack::<W, B>(
                                array_ref![input, 0, B],
                                reference,
                                array_mut_ref![output, 0, 1024],
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        Self::T => {
                            const W: usize = <$T>::T;
                            const B: usize = 1024;
                            Self::unfor_pack::<W, B>(
                                array_ref![input, 0, 1024],
                                reference,
                                array_mut_ref![output, 0, 1024],
                            )
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }
        }
    };
}

impl_for!(u8);
impl_for!(u16);
impl_for!(u32);
impl_for!(u64);

#[cfg(test)]
mod test {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn test_ffor() {
        const W: usize = 15;
        const B: usize = 1024 * W / u16::T;

        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << W)) as u16 + 10;
        }

        let mut packed = [0; 128 * W / size_of::<u16>()];
        FoR::for_pack::<W, B>(&values, 10, &mut packed);

        let mut unpacked = [0; 1024];
        BitPacking::unpack::<W, B>(&packed, &mut unpacked);

        for (i, (a, b)) in values.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(
                // Check that the unpacked array is 10 less than the original (modulo 2^15)
                a.wrapping_sub(10) & ((1 << W) - 1),
                *b,
                "Mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_ffor_unchecked() {
        const W: usize = 15;
        const B: usize = 1024 * W / u16::T;

        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << W)) as u16 + 10;
        }

        let mut packed = [0; 128 * W / size_of::<u16>()];
        FoR::for_pack::<W, B>(&values, 10, &mut packed);

        let mut unpacked = [0; 1024];
        unsafe {
            FoR::unchecked_unfor_pack(W, &packed, 10, &mut unpacked);
        }

        for (i, (a, b)) in values.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(
                // Check that the unpacked array is 10 less than the original (modulo 2^15)
                *a,
                *b,
                "Mismatch at index {i}"
            );
        }
    }
}
