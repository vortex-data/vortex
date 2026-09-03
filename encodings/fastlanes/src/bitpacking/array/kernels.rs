// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Width-resolved FastLanes kernels for bit-packed arrays.
//!
//! The FastLanes kernels are generic over the packed bit width `W`, so the runtime-width
//! `unchecked_*` entry points of the `fastlanes` crate dispatch on the width with a `match` on
//! every call. A [`BitPackedArray`](crate::BitPackedArray) knows its width but is type erased, so
//! it cannot name the instantiation statically. Instead, the array resolves function pointers to
//! the concrete instantiations once, when it is constructed, and hands them out as
//! [`BitPackedKernels`]. The decoding paths then call the resolved kernels block after block
//! without re-dispatching.

use std::mem;

use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
use fastlanes::FoR;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

/// Packs one FastLanes block of 1024 values.
///
/// # Safety
///
/// `input` must hold exactly 1024 elements and `output` exactly `128 * bit_width / size_of::<P>()`.
/// The lengths are checked only with `debug_assert`.
pub type PackFn<P> = unsafe fn(input: &[P], output: &mut [P]);

/// Unpacks one FastLanes block of 1024 values.
///
/// # Safety
///
/// `packed` must hold exactly [`BitPackedKernels::packed_block_len`] elements and `output`
/// exactly 1024. The lengths are checked only with `debug_assert`.
pub type UnpackFn<P> = unsafe fn(packed: &[P], output: &mut [P]);

/// Unpacks the value at `index` of one packed FastLanes block.
///
/// # Safety
///
/// `packed` must hold exactly [`BitPackedKernels::packed_block_len`] elements. The length is
/// checked only with `debug_assert`. An `index` of 1024 or more panics.
pub type UnpackSingleFn<P> = unsafe fn(packed: &[P], index: usize) -> P;

/// Unpacks one FastLanes block and wrapping-adds `reference` to every value.
///
/// # Safety
///
/// `packed` must hold exactly [`BitPackedKernels::packed_block_len`] elements and `output`
/// exactly 1024. The lengths are checked only with `debug_assert`.
pub type UnforPackFn<P> = unsafe fn(packed: &[P], reference: P, output: &mut [P]);

/// Unpacks one FastLanes block, comparing each value against `rhs` with `cmp` and writing the
/// results as a lane-major 1024-bit mask. See [`BitPackingCompare::unpack_cmp`] for the layout.
///
/// # Safety
///
/// `packed` must hold exactly `128 * bit_width / size_of::<P>()` elements. The length is checked
/// only with `debug_assert`.
pub type UnpackCmpFn<P, V, F> = unsafe fn(packed: &[P], output: &mut [u64; 16], cmp: F, rhs: V);

/// FastLanes kernels resolved for one bit width of physical type `P`.
///
/// Obtained from [`BitPackedData::kernels`](crate::BitPackedData::kernels). Each kernel is the
/// const-width instantiation for the array's bit width, so calling it does not dispatch on the
/// width.
#[derive(Clone, Copy, Debug)]
pub struct BitPackedKernels<P> {
    /// The unsigned [`PType`] of `P`, kept so the type-erased form can check it before
    /// re-typing the pointers.
    ptype: PType,
    bit_width: u8,
    /// Unpacks one packed block into 1024 values.
    pub unpack: UnpackFn<P>,
    /// Unpacks a single value of one packed block.
    pub unpack_single: UnpackSingleFn<P>,
    /// Unpacks one packed block, adding a frame-of-reference value.
    pub unfor_pack: UnforPackFn<P>,
}

impl<P> BitPackedKernels<P> {
    /// The bit width the kernels were resolved for.
    #[inline]
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }
}

impl<P: BitPackedPhysical> BitPackedKernels<P> {
    /// The number of `P` elements holding one packed block of 1024 values.
    #[inline]
    pub fn packed_block_len(&self) -> usize {
        128 * self.bit_width as usize / size_of::<P>()
    }
}

/// [`BitPackedKernels`] with the physical type erased, as stored by
/// [`BitPackedData`](crate::BitPackedData).
///
/// The function pointers are those of a `BitPackedKernels<P>` for the recorded `ptype`, cast to
/// a placeholder element type. [`Self::typed`] casts them back; they are never called in the
/// erased form.
pub type ResolvedKernels = BitPackedKernels<()>;

impl ResolvedKernels {
    /// Resolves the kernels for an array of `ptype` packed to `bit_width` bits.
    ///
    /// Signed types resolve to their unsigned counterpart, which is what the packed buffer holds.
    pub fn try_new(ptype: PType, bit_width: u8) -> VortexResult<Self> {
        vortex_ensure!(ptype.is_int(), MismatchedTypes: "integer", ptype);
        vortex_ensure!(
            bit_width as usize <= ptype.bit_width(),
            "Unsupported bit width {bit_width} for {ptype}"
        );
        Ok(match_each_unsigned_integer_ptype!(
            ptype.to_unsigned(),
            |P| { P::resolve_kernels(bit_width).erase() }
        ))
    }

    /// The kernels typed for `P`, which must be the physical type they were resolved for.
    ///
    /// # Panics
    ///
    /// If `P` is not the physical type the kernels were resolved for.
    #[inline]
    pub fn typed<P: BitPackedPhysical>(self) -> BitPackedKernels<P> {
        assert!(
            self.ptype == P::PTYPE,
            "BitPacked kernels were resolved for a different physical type"
        );
        // SAFETY: `ptype` records the `P` these pointers were erased from (see `erase`), and
        // transmuting a function pointer back to its original signature is lossless.
        unsafe {
            BitPackedKernels {
                ptype: self.ptype,
                bit_width: self.bit_width,
                unpack: mem::transmute::<UnpackFn<()>, UnpackFn<P>>(self.unpack),
                unpack_single: mem::transmute::<UnpackSingleFn<()>, UnpackSingleFn<P>>(
                    self.unpack_single,
                ),
                unfor_pack: mem::transmute::<UnforPackFn<()>, UnforPackFn<P>>(self.unfor_pack),
            }
        }
    }
}

impl<P: BitPackedPhysical> BitPackedKernels<P> {
    /// Erases `P` from the function pointers; [`ResolvedKernels::typed`] restores it.
    fn erase(self) -> ResolvedKernels {
        // SAFETY: Function pointers of every signature share one layout, and the erased pointers
        // are only ever called after `typed` casts them back to this signature, which `ptype`
        // enforces.
        unsafe {
            BitPackedKernels {
                ptype: self.ptype,
                bit_width: self.bit_width,
                unpack: mem::transmute::<UnpackFn<P>, UnpackFn<()>>(self.unpack),
                unpack_single: mem::transmute::<UnpackSingleFn<P>, UnpackSingleFn<()>>(
                    self.unpack_single,
                ),
                unfor_pack: mem::transmute::<UnforPackFn<P>, UnforPackFn<()>>(self.unfor_pack),
            }
        }
    }
}

/// The physical storage types of a bit-packed array, i.e. the unsigned integers the FastLanes
/// kernels are implemented for. Signed arrays are packed as their unsigned counterpart.
pub trait BitPackedPhysical: NativePType + BitPacking + BitPackingCompare + FoR {
    /// Resolves the kernels for `bit_width`, which must not exceed the width of `Self`.
    fn resolve_kernels(bit_width: u8) -> BitPackedKernels<Self>;

    /// Resolves the pack kernel for `bit_width`, which must not exceed the width of `Self`.
    ///
    /// Packing happens before an array exists to cache kernels on, so callers resolve this once
    /// per buffer instead.
    fn resolve_pack(bit_width: u8) -> PackFn<Self>;

    /// Resolves the fused unpack-and-compare kernel for `bit_width`, which must not exceed the
    /// width of `Self`.
    fn resolve_unpack_cmp<V, F>(bit_width: u8) -> UnpackCmpFn<Self, V, F>
    where
        V: FastLanesComparable<Bitpacked = Self>,
        F: Fn(V, V) -> bool;
}

unsafe fn pack<P: BitPacking, const W: usize, const B: usize>(input: &[P], output: &mut [P]) {
    // SAFETY: The caller upholds the `PackFn` length contract.
    unsafe { P::pack::<W, B>(as_block(input), as_block_mut(output)) }
}

unsafe fn unpack<P: BitPacking, const W: usize, const B: usize>(packed: &[P], output: &mut [P]) {
    // SAFETY: The caller upholds the `UnpackFn` length contract.
    unsafe { P::unpack::<W, B>(as_block(packed), as_block_mut(output)) }
}

unsafe fn unpack_single<P: BitPacking, const W: usize, const B: usize>(
    packed: &[P],
    index: usize,
) -> P {
    // SAFETY: The caller upholds the `UnpackSingleFn` length contract.
    P::unpack_single::<W, B>(unsafe { as_block(packed) }, index)
}

unsafe fn unfor_pack<P: FoR, const W: usize, const B: usize>(
    packed: &[P],
    reference: P,
    output: &mut [P],
) {
    // SAFETY: The caller upholds the `UnforPackFn` length contract.
    unsafe { P::unfor_pack::<W, B>(as_block(packed), reference, as_block_mut(output)) }
}

unsafe fn unpack_cmp<P: BitPackingCompare, const W: usize, const B: usize, V, F>(
    packed: &[P],
    output: &mut [u64; 16],
    cmp: F,
    rhs: V,
) where
    V: FastLanesComparable<Bitpacked = P>,
    F: Fn(V, V) -> bool,
{
    // SAFETY: The caller upholds the `UnpackCmpFn` length contract.
    P::unpack_cmp::<W, B, V, F>(unsafe { as_block(packed) }, output, cmp, rhs);
}

/// Reinterprets `slice` as a block of exactly `N` elements.
///
/// # Safety
///
/// `slice.len()` must be `N`. This is checked only with `debug_assert`.
unsafe fn as_block<P, const N: usize>(slice: &[P]) -> &[P; N] {
    debug_assert_eq!(slice.len(), N);
    // SAFETY: The caller guarantees `N` elements, and `[P; N]` has the alignment of `P`.
    unsafe { &*slice.as_ptr().cast::<[P; N]>() }
}

/// Reinterprets `slice` as a mutable block of exactly `N` elements.
///
/// # Safety
///
/// `slice.len()` must be `N`. This is checked only with `debug_assert`.
unsafe fn as_block_mut<P, const N: usize>(slice: &mut [P]) -> &mut [P; N] {
    debug_assert_eq!(slice.len(), N);
    // SAFETY: The caller guarantees `N` elements, and `[P; N]` has the alignment of `P`.
    unsafe { &mut *slice.as_mut_ptr().cast::<[P; N]>() }
}

macro_rules! impl_bitpacked_physical {
    ($P:ty, $bits:literal) => {
        impl BitPackedPhysical for $P {
            fn resolve_kernels(bit_width: u8) -> BitPackedKernels<Self> {
                seq_macro::seq!(W in 0..=$bits {
                    match bit_width {
                        #(W => BitPackedKernels {
                            ptype: <$P as NativePType>::PTYPE,
                            bit_width,
                            unpack: unpack::<$P, W, { 1024 * W / $bits }>,
                            unpack_single: unpack_single::<$P, W, { 1024 * W / $bits }>,
                            unfor_pack: unfor_pack::<$P, W, { 1024 * W / $bits }>,
                        },)*
                        _ => vortex_panic!(
                            "Unsupported bit width {bit_width} for {}",
                            <$P as NativePType>::PTYPE
                        ),
                    }
                })
            }

            fn resolve_pack(bit_width: u8) -> PackFn<Self> {
                seq_macro::seq!(W in 0..=$bits {
                    match bit_width {
                        #(W => pack::<$P, W, { 1024 * W / $bits }>,)*
                        _ => vortex_panic!(
                            "Unsupported bit width {bit_width} for {}",
                            <$P as NativePType>::PTYPE
                        ),
                    }
                })
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

impl_bitpacked_physical!(u8, 8);
impl_bitpacked_physical!(u16, 16);
impl_bitpacked_physical!(u32, 32);
impl_bitpacked_physical!(u64, 64);

#[cfg(test)]
mod tests {
    use num_traits::WrappingAdd;
    use rstest::rstest;

    use super::*;

    /// Every width of every physical type resolves to kernels that agree with the runtime-width
    /// FastLanes entry points, both directly and after a round trip through the erased form.
    fn assert_kernels_match_fastlanes<P>()
    where
        P: BitPackedPhysical + WrappingAdd + FastLanesComparable<Bitpacked = P>,
    {
        let values: [P; 1024] = std::array::from_fn(|i| P::from(i % 251).unwrap());
        for bit_width in 0..=(8 * size_of::<P>() as u8) {
            let kernels = ResolvedKernels::try_new(P::PTYPE, bit_width)
                .unwrap()
                .typed::<P>();
            assert_eq!(kernels.bit_width(), bit_width);
            assert_eq!(
                kernels.unpack as usize,
                P::resolve_kernels(bit_width).unpack as usize,
                "erased round trip at width {bit_width}"
            );

            let block_len = kernels.packed_block_len();
            let mut expected_packed = vec![P::zero(); block_len];
            // SAFETY: `expected_packed` holds exactly one block at `bit_width` and `values` 1024
            // values.
            unsafe { P::unchecked_pack(bit_width as usize, &values, &mut expected_packed) };

            let mut packed = vec![P::zero(); block_len];
            // SAFETY: `values` holds 1024 values and `packed` exactly one block at `bit_width`.
            unsafe { P::resolve_pack(bit_width)(&values, &mut packed) };
            assert_eq!(packed, expected_packed, "pack at width {bit_width}");

            let mut expected = [P::zero(); 1024];
            // SAFETY: `packed` holds exactly one block at `bit_width` and `expected` 1024 values.
            unsafe { P::unchecked_unpack(bit_width as usize, &packed, &mut expected) };

            // SAFETY: `packed` holds exactly one block at `bit_width` and `unpacked` 1024 values.
            let mut unpacked = [P::zero(); 1024];
            unsafe { (kernels.unpack)(&packed, &mut unpacked) };
            assert_eq!(unpacked, expected, "unpack at width {bit_width}");

            for index in [0, 1, 511, 1023] {
                assert_eq!(
                    // SAFETY: `packed` holds exactly one block at `bit_width`.
                    unsafe { (kernels.unpack_single)(&packed, index) },
                    expected[index],
                    "unpack_single at width {bit_width} index {index}"
                );
            }

            let reference = P::from(7).unwrap();
            let mut unfor = [P::zero(); 1024];
            // SAFETY: `packed` holds exactly one block at `bit_width` and `unfor` 1024 values.
            unsafe { (kernels.unfor_pack)(&packed, reference, &mut unfor) };
            for (got, want) in unfor.iter().zip(expected) {
                assert_eq!(
                    *got,
                    want.wrapping_add(&reference),
                    "unfor_pack at {bit_width}"
                );
            }

            let rhs = P::from(100).unwrap();
            let mut mask = [0u64; 16];
            // SAFETY: `packed` holds exactly one block at `bit_width`.
            unsafe {
                P::resolve_unpack_cmp::<P, _>(bit_width)(&packed, &mut mask, |a, b| a < b, rhs)
            };
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
    fn signed_resolves_to_unsigned_kernels() {
        let resolved = ResolvedKernels::try_new(PType::I16, 3).unwrap();
        assert_eq!(
            resolved.typed::<u16>().unpack as usize,
            u16::resolve_kernels(3).unpack as usize
        );
    }

    #[test]
    fn rejects_width_beyond_type() {
        assert!(ResolvedKernels::try_new(PType::U8, 9).is_err());
        assert!(ResolvedKernels::try_new(PType::F32, 3).is_err());
        assert!(ResolvedKernels::try_new(PType::U8, 8).is_ok());
    }

    #[test]
    #[should_panic(expected = "resolved for a different physical type")]
    fn typed_rejects_other_types() {
        ResolvedKernels::try_new(PType::U16, 3)
            .unwrap()
            .typed::<u8>();
    }
}
