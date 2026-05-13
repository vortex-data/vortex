// SPDX-FileCopyrightText: Copyright the FastLanes Authors
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0
//
// Vendored verbatim from `fastlanes` 0.5.0 / src/bitpacking.rs.

use arrayref::{array_mut_ref, array_ref};
use const_for::const_for;
use core::mem::size_of;
use paste::paste;

use crate::{pack, seq_t, supported_bit_width, unpack, FastLanes, FL_ORDER};

/// `BitPack` into a compile-time known bit-width.
pub trait BitPacking: FastLanes {
    /// Packs 1024 elements into W bits each.
    /// The output is given as Self to ensure correct alignment.
    fn pack<const W: usize, const B: usize>(input: &[Self; 1024], output: &mut [Self; B]);

    /// Packs 1024 elements into `W` bits each, where `W` is runtime-known instead of
    /// compile-time known.
    ///
    /// # Safety
    /// The input slice must be of exactly length 1024. The output slice must be of length
    /// `1024 * W / T`, where `T` is the bit-width of Self and `W` is the packed width.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_pack(width: usize, input: &[Self], output: &mut [Self]);

    /// Unpacks 1024 elements from `W` bits each.
    fn unpack<const W: usize, const B: usize>(input: &[Self; B], output: &mut [Self; 1024]);

    /// Unpacks 1024 elements from `W` bits each, where `W` is runtime-known instead of
    /// compile-time known.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output slice must be of exactly length 1024.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack(width: usize, input: &[Self], output: &mut [Self]);

    /// Unpacks a single element at the provided index from a packed array of 1024 `W` bit elements.
    fn unpack_single<const W: usize, const B: usize>(packed: &[Self; B], index: usize) -> Self;

    /// Unpacks a single element at the provided index from a packed array of 1024 `W` bit elements,
    /// where `W` is runtime-known instead of compile-time known.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output slice must be of exactly length 1024.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack_single(width: usize, input: &[Self], index: usize) -> Self;
}

macro_rules! impl_packing {
    ($T:ty) => {
        impl BitPacking for $T {
            #[inline(never)]
            fn pack<const W: usize, const B: usize>(
                input: &[Self; 1024],
                output: &mut [Self; B],
            ) {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }


                for lane in 0..Self::LANES {
                    pack!($T, W, output, lane, |$idx| {
                        input[$idx]
                    });
                }
            }

            unsafe fn unchecked_pack(width: usize, input: &[Self], output: &mut [Self]) {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(output.len(), packed_len, "Output buffer must be of size 1024 * W / T");
                debug_assert_eq!(input.len(), 1024, "Input buffer must be of size 1024");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::pack::<W, B>(
                                array_ref![input, 0, 1024],
                                array_mut_ref![output, 0, B],
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        Self::T => {
                            // How large is the target buffer size?
                            const W: usize = <$T>::T;
                            const B: usize = 1024;
                            Self::pack::<W, B>(
                                array_ref![input, 0, 1024],
                                array_mut_ref![output, 0, 1024],
                            )
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }

            #[inline(never)]
            fn unpack<const W: usize, const B: usize>(
                input: &[Self; B],
                output: &mut [Self; 1024],
            ) {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }


                for lane in 0..Self::LANES {
                    unpack!($T, W, input, lane, |$idx, $elem| {
                        output[$idx] = $elem
                    });
                }
            }

            unsafe fn unchecked_unpack(width: usize, input: &[Self], output: &mut [Self]) {
                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
                debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / <$T>::T;
                            Self::unpack::<W, B>(
                                array_ref![input, 0, B],
                                array_mut_ref![output, 0, 1024],
                            )
                        },)*
                        // seq_t has exclusive upper bound
                        Self::T => {
                            const W: usize = <$T>::T;
                            const B: usize = 1024;
                            Self::unpack::<W, B>(
                                array_ref![input, 0, 1024],
                                array_mut_ref![output, 0, 1024],
                            )
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }

            /// Unpacks a single element at the provided index from a packed array of 1024 `W` bit elements.
            fn unpack_single<const W: usize, const B: usize>(packed: &[Self; B], index: usize) -> Self
            {
                const {
                    assert!(supported_bit_width(W, 8 * core::mem::size_of::<$T>()));
                    assert!(B == 1024 * W / Self::T);
                }

                if W == 0 {
                    // Special case for W=0, we just need to zero the output.
                    return 0 as $T;
                }

                // We can think of the input array as effectively a row-major, left-to-right
                // 2-D array of with `Self::LANES` columns and `Self::T` rows.
                //
                // Meanwhile, we can think of the packed array as either:
                //      1. `Self::T` rows of W-bit elements, with `Self::LANES` columns
                //      2. `W` rows of `Self::T`-bit words, with `Self::LANES` columns
                //
                // Bitpacking involves a transposition of the input array ordering, such that
                // decompression can be fused efficiently with encodings like delta and RLE.
                //
                // First step, we need to get the lane and row for interpretation #1 above.
                assert!(index < 1024, "Index must be less than 1024, got {}", index);
                let (lane, row): (usize, usize) = {
                    const LANES: [u8; 1024] = lanes_by_index::<$T>();
                    const ROWS: [u8; 1024] = rows_by_index::<$T>();
                    (LANES[index] as usize, ROWS[index] as usize)
                };

                if W == <$T>::T {
                    // Special case for W==T, we can just read the value directly
                    return packed[<$T>::LANES * row + lane];
                }

                let mask: $T = (1 << (W % <$T>::T)) - 1;
                let start_bit = row * W;
                let start_word = start_bit / <$T>::T;
                let lo_shift = start_bit % <$T>::T;
                let remaining_bits = <$T>::T - lo_shift;

                let lo = packed[<$T>::LANES * start_word + lane] >> lo_shift;
                return if remaining_bits >= W {
                    // in this case we will mask out all bits of hi word
                    lo & mask
                } else {
                    // guaranteed that lo_shift > 0 and thus remaining_bits < T
                    let hi = packed[<$T>::LANES * (start_word + 1) + lane] << remaining_bits;
                    (lo | hi) & mask
                };
            }

            unsafe fn unchecked_unpack_single(width: usize, packed: &[Self], index: usize) -> Self {
                const T: usize = <$T>::T;

                let packed_len = 128 * width / size_of::<Self>();
                debug_assert_eq!(packed.len(), packed_len, "Input buffer must be of size {}", packed_len);
                debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

                paste!(seq_t!(W in $T {
                    match width {
                        #(W => {
                            const B: usize = 1024 * W / T;
                            return <$T>::unpack_single::<W, B>(array_ref![packed, 0, B], index);
                        },)*
                        // seq_t has exclusive upper bound
                        T => {
                            const W: usize = T;
                            const B: usize = 1024;
                            return <$T>::unpack_single::<W, B>(array_ref![packed, 0, 1024], index);
                        },
                        _ => unreachable!("Unsupported width: {}", width)
                    }
                }))
            }
        }
    };
}

// helper function executed at compile-time to speed up unpack_single at runtime
const fn lanes_by_index<T: FastLanes>() -> [u8; 1024] {
    let mut lanes = [0u8; 1024];
    const_for!(i in 0..1024 => {
        lanes[i] = (i % T::LANES) as u8;
    });
    lanes
}

// helper function executed at compile-time to speed up unpack_single at runtime
const fn rows_by_index<T: FastLanes>() -> [u8; 1024] {
    let mut rows = [0u8; 1024];
    const_for!(i in 0..1024 => {
        // This is the inverse of the `index` function from the pack/unpack macros:
        //     fn index(row: usize, lane: usize) -> usize {
        //         let o = row / 8;
        //         let s = row % 8;
        //         (FL_ORDER[o] * 16) + (s * 128) + lane
        //     }
        let lane = i % T::LANES;
        let s = i / 128; // because `(FL_ORDER[o] * 16) + lane` is always < 128
        let fl_order = (i - s * 128 - lane) / 16; // value of FL_ORDER[o]
        let o = FL_ORDER[fl_order]; // because this transposition is invertible!
        rows[i] = (o * 8 + s) as u8;
    });
    rows
}

impl_packing!(u8);
impl_packing!(u16);
impl_packing!(u32);
impl_packing!(u64);

#[cfg(test)]
mod test {
    use core::array;
    use core::fmt::Debug;
    use seq_macro::seq;

    use super::*;

    #[test]
    fn test_unchecked_pack() {
        let input = array::from_fn(|i| i as u32);
        let mut packed = [0; 320];
        unsafe { BitPacking::unchecked_pack(10, &input, &mut packed) };
        let mut output = [0; 1024];
        unsafe { BitPacking::unchecked_unpack(10, &packed, &mut output) };
        assert_eq!(input, output);
    }

    #[test]
    fn test_unpack_single() {
        let values = array::from_fn(|i| i as u32);
        let mut packed = [0; 512];
        BitPacking::pack::<16, 512>(&values, &mut packed);

        for i in 0..1024 {
            assert_eq!(BitPacking::unpack_single::<16, 512>(&packed, i), values[i]);
            assert_eq!(
                unsafe { BitPacking::unchecked_unpack_single(16, &packed, i) },
                values[i]
            );
        }
    }

    fn try_round_trip<T: BitPacking + Debug, const W: usize, const B: usize>() {
        let mut values: [T; 1024] = [T::zero(); 1024];
        for i in 0..1024 {
            values[i] = T::from(i % (1 << (W % T::T))).unwrap();
        }

        let mut packed = [T::zero(); B];
        BitPacking::pack::<W, B>(&values, &mut packed);

        let mut unpacked = [T::zero(); 1024];
        BitPacking::unpack::<W, B>(&packed, &mut unpacked);

        assert_eq!(&unpacked, &values);

        for i in 0..1024 {
            assert_eq!(BitPacking::unpack_single::<W, B>(&packed, i), values[i]);
            assert_eq!(
                unsafe { BitPacking::unchecked_unpack_single(W, &packed, i) },
                values[i]
            );
        }
    }

    macro_rules! impl_try_round_trip {
        ($T:ty, $W:expr) => {
            paste! {
                #[test]
                fn [<test_round_trip_ $T _ $W>]() {
                    const B: usize = 1024 * $W / <$T>::T;
                    try_round_trip::<$T, $W, B>();
                }
            }
        };
    }

    seq!(W in 0..=8 { impl_try_round_trip!(u8, W); });
    seq!(W in 0..=16 { impl_try_round_trip!(u16, W); });
    seq!(W in 0..=32 { impl_try_round_trip!(u32, W); });
    seq!(W in 0..=64 { impl_try_round_trip!(u64, W); });
}
