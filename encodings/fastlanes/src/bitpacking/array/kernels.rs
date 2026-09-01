// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Width-resolved FastLanes kernels for bit-packed arrays.
//!
//! The FastLanes kernels are generic over the packed bit width `W`, so the runtime-width
//! `unchecked_*` entry points of the `fastlanes` crate dispatch on the width with a `match` on
//! every call. A [`BitPackedArray`](crate::BitPackedArray) knows its width but is type erased, so
//! it cannot name the instantiation statically. Instead, the array resolves function pointers to
//! the concrete instantiations once, lazily, and hands them out as [`BitPackedKernels`]. The
//! decoding paths then call the resolved kernels block after block without re-dispatching.

use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
use fastlanes::FoR;
use vortex_array::dtype::NativePType;
use vortex_error::vortex_panic;

/// Unpacks one FastLanes block of 1024 values.
///
/// `packed` must hold exactly [`BitPackedKernels::packed_block_len`] elements and `output`
/// exactly 1024.
pub type UnpackFn<P> = fn(packed: &[P], output: &mut [P]);

/// Unpacks the value at `index` (`< 1024`) of one packed FastLanes block.
///
/// `packed` must hold exactly [`BitPackedKernels::packed_block_len`] elements.
pub type UnpackSingleFn<P> = fn(packed: &[P], index: usize) -> P;

/// Unpacks one FastLanes block and wrapping-adds `reference` to every value.
///
/// `packed` must hold exactly [`BitPackedKernels::packed_block_len`] elements and `output`
/// exactly 1024.
pub type UnforPackFn<P> = fn(packed: &[P], reference: P, output: &mut [P]);

/// Unpacks one FastLanes block, comparing each value against `rhs` with `cmp` and writing the
/// results as a lane-major 1024-bit mask. See [`BitPackingCompare::unpack_cmp`] for the layout.
///
/// `packed` must hold exactly `128 * bit_width / size_of::<P>()` elements.
pub type UnpackCmpFn<P, V, F> = fn(packed: &[P], output: &mut [u64; 16], cmp: F, rhs: V);

/// FastLanes kernels resolved for one bit width of physical type `P`.
///
/// Obtained from [`BitPackedData::kernels`](crate::BitPackedData::kernels). Each kernel is the
/// const-width instantiation for the array's bit width, so calling it does not dispatch on the
/// width. The kernels check the block lengths they are handed and panic on a mismatch.
#[derive(Clone, Copy, Debug)]
pub struct BitPackedKernels<P> {
    bit_width: u8,
    /// Unpacks one packed block into 1024 values.
    pub unpack: UnpackFn<P>,
    /// Unpacks a single value of one packed block.
    pub unpack_single: UnpackSingleFn<P>,
    /// Unpacks one packed block, adding a frame-of-reference value.
    pub unfor_pack: UnforPackFn<P>,
}

impl<P: BitPackedPhysical> BitPackedKernels<P> {
    /// The bit width the kernels were resolved for.
    #[inline]
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// The number of `P` elements holding one packed block of 1024 values.
    #[inline]
    pub fn packed_block_len(&self) -> usize {
        128 * self.bit_width as usize / size_of::<P>()
    }
}

/// [`BitPackedKernels`] resolved for one of the physical types, stored type erased by
/// [`BitPackedData`](crate::BitPackedData).
#[derive(Clone, Copy, Debug)]
pub enum ResolvedKernels {
    U8(BitPackedKernels<u8>),
    U16(BitPackedKernels<u16>),
    U32(BitPackedKernels<u32>),
    U64(BitPackedKernels<u64>),
}

/// The physical storage types of a bit-packed array, i.e. the unsigned integers the FastLanes
/// kernels are implemented for. Signed arrays are packed as their unsigned counterpart.
pub trait BitPackedPhysical: NativePType + BitPacking + BitPackingCompare + FoR {
    /// Resolves the kernels for `bit_width`, which must not exceed the width of `Self`.
    fn resolve_kernels(bit_width: u8) -> ResolvedKernels;

    /// Returns the kernels if `resolved` holds kernels for `Self`.
    fn kernels_from(resolved: &ResolvedKernels) -> Option<BitPackedKernels<Self>>;

    /// Resolves the fused unpack-and-compare kernel for `bit_width`, which must not exceed the
    /// width of `Self`.
    fn resolve_unpack_cmp<V, F>(bit_width: u8) -> UnpackCmpFn<Self, V, F>
    where
        V: FastLanesComparable<Bitpacked = Self>,
        F: Fn(V, V) -> bool;
}

fn unpack<P: BitPacking, const W: usize, const B: usize>(packed: &[P], output: &mut [P]) {
    P::unpack::<W, B>(as_block(packed), as_block_mut(output));
}

fn unpack_single<P: BitPacking, const W: usize, const B: usize>(packed: &[P], index: usize) -> P {
    P::unpack_single::<W, B>(as_block(packed), index)
}

fn unfor_pack<P: FoR, const W: usize, const B: usize>(
    packed: &[P],
    reference: P,
    output: &mut [P],
) {
    P::unfor_pack::<W, B>(as_block(packed), reference, as_block_mut(output));
}

fn unpack_cmp<P: BitPackingCompare, const W: usize, const B: usize, V, F>(
    packed: &[P],
    output: &mut [u64; 16],
    cmp: F,
    rhs: V,
) where
    V: FastLanesComparable<Bitpacked = P>,
    F: Fn(V, V) -> bool,
{
    P::unpack_cmp::<W, B, V, F>(as_block(packed), output, cmp, rhs);
}

#[inline(always)]
fn as_block<P, const N: usize>(slice: &[P]) -> &[P; N] {
    match slice.try_into() {
        Ok(block) => block,
        Err(_) => vortex_panic!(
            "Expected a FastLanes block of {N} elements, got {}",
            slice.len()
        ),
    }
}

#[inline(always)]
fn as_block_mut<P, const N: usize>(slice: &mut [P]) -> &mut [P; N] {
    let len = slice.len();
    match slice.try_into() {
        Ok(block) => block,
        Err(_) => vortex_panic!("Expected a FastLanes block of {N} elements, got {len}"),
    }
}

macro_rules! impl_bitpacked_physical {
    ($P:ty, $variant:ident, $bits:literal) => {
        impl BitPackedPhysical for $P {
            fn resolve_kernels(bit_width: u8) -> ResolvedKernels {
                seq_macro::seq!(W in 0..=$bits {
                    match bit_width {
                        #(W => ResolvedKernels::$variant(BitPackedKernels {
                            bit_width,
                            unpack: unpack::<$P, W, { 1024 * W / $bits }>,
                            unpack_single: unpack_single::<$P, W, { 1024 * W / $bits }>,
                            unfor_pack: unfor_pack::<$P, W, { 1024 * W / $bits }>,
                        }),)*
                        _ => vortex_panic!(
                            "Unsupported bit width {bit_width} for {}",
                            <$P as NativePType>::PTYPE
                        ),
                    }
                })
            }

            fn kernels_from(resolved: &ResolvedKernels) -> Option<BitPackedKernels<Self>> {
                match resolved {
                    ResolvedKernels::$variant(kernels) => Some(*kernels),
                    _ => None,
                }
            }

            fn resolve_unpack_cmp<V, F>(bit_width: u8) -> UnpackCmpFn<Self, V, F>
            where
                V: FastLanesComparable<Bitpacked = Self>,
                F: Fn(V, V) -> bool,
            {
                seq_macro::seq!(W in 0..=$bits {
                    match bit_width {
                        #(W => unpack_cmp::<$P, W, { 1024 * W / $bits }, V, F>,)*
                        _ => vortex_panic!(
                            "Unsupported bit width {bit_width} for {}",
                            <$P as NativePType>::PTYPE
                        ),
                    }
                })
            }
        }
    };
}

impl_bitpacked_physical!(u8, U8, 8);
impl_bitpacked_physical!(u16, U16, 16);
impl_bitpacked_physical!(u32, U32, 32);
impl_bitpacked_physical!(u64, U64, 64);

#[cfg(test)]
mod tests {
    use num_traits::WrappingAdd;
    use rstest::rstest;

    use super::*;

    /// Every width of every physical type resolves to kernels that agree with the runtime-width
    /// FastLanes entry points.
    fn assert_kernels_match_fastlanes<P>()
    where
        P: BitPackedPhysical + WrappingAdd + FastLanesComparable<Bitpacked = P>,
    {
        let values: [P; 1024] = std::array::from_fn(|i| P::from(i % 251).unwrap());
        for bit_width in 0..=(8 * size_of::<P>() as u8) {
            let kernels = P::kernels_from(&P::resolve_kernels(bit_width)).unwrap();
            assert_eq!(kernels.bit_width(), bit_width);

            let block_len = kernels.packed_block_len();
            let mut packed = vec![P::zero(); block_len];
            // SAFETY: `packed` holds exactly one block at `bit_width` and `values` 1024 values.
            unsafe { P::unchecked_pack(bit_width as usize, &values, &mut packed) };

            let mut expected = [P::zero(); 1024];
            // SAFETY: `packed` holds exactly one block at `bit_width` and `expected` 1024 values.
            unsafe { P::unchecked_unpack(bit_width as usize, &packed, &mut expected) };

            let mut unpacked = [P::zero(); 1024];
            (kernels.unpack)(&packed, &mut unpacked);
            assert_eq!(unpacked, expected, "unpack at width {bit_width}");

            for index in [0, 1, 511, 1023] {
                assert_eq!(
                    (kernels.unpack_single)(&packed, index),
                    expected[index],
                    "unpack_single at width {bit_width} index {index}"
                );
            }

            let reference = P::from(7).unwrap();
            let mut unfor = [P::zero(); 1024];
            (kernels.unfor_pack)(&packed, reference, &mut unfor);
            for (got, want) in unfor.iter().zip(expected) {
                assert_eq!(
                    *got,
                    want.wrapping_add(&reference),
                    "unfor_pack at {bit_width}"
                );
            }

            let rhs = P::from(100).unwrap();
            let mut mask = [0u64; 16];
            P::resolve_unpack_cmp::<P, _>(bit_width)(&packed, &mut mask, |a, b| a < b, rhs);
            let mut expected_mask = [0u64; 16];
            // SAFETY: `packed` holds exactly one block at `bit_width`.
            unsafe {
                P::unchecked_unpack_cmp::<P, _>(
                    bit_width as usize,
                    &packed,
                    &mut expected_mask,
                    |a, b| a < b,
                    rhs,
                );
            }
            assert_eq!(mask, expected_mask, "unpack_cmp at width {bit_width}");
        }
    }

    #[rstest]
    #[case::u8(assert_kernels_match_fastlanes::<u8>)]
    #[case::u16(assert_kernels_match_fastlanes::<u16>)]
    #[case::u32(assert_kernels_match_fastlanes::<u32>)]
    #[case::u64(assert_kernels_match_fastlanes::<u64>)]
    fn kernels_match_fastlanes(#[case] check: fn()) {
        check();
    }

    #[test]
    fn kernels_from_rejects_other_types() {
        let resolved = u16::resolve_kernels(3);
        assert!(u16::kernels_from(&resolved).is_some());
        assert!(u8::kernels_from(&resolved).is_none());
        assert!(u32::kernels_from(&resolved).is_none());
    }

    #[test]
    #[should_panic(expected = "Expected a FastLanes block of 1024 elements")]
    fn unpack_rejects_short_output() {
        let kernels = u8::kernels_from(&u8::resolve_kernels(1)).unwrap();
        let packed = [0u8; 128];
        let mut output = [0u8; 512];
        (kernels.unpack)(&packed, &mut output);
    }
}
