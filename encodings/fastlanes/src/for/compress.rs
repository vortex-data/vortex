// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::FoR;
use num_traits::{PrimInt, WrappingAdd, WrappingSub};
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::stats::Stat;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{IntoArray, ToCanonical};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{
    NativePType, PhysicalPType, UnsignedPType, match_each_integer_ptype,
    match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::unpack_iter::{UnpackStrategy, UnpackedChunks};
use crate::{BitPackedArray, BitPackedVTable, FoRArray, bitpack_decompress};

impl FoRArray {
    pub fn encode(array: PrimitiveArray) -> VortexResult<FoRArray> {
        let min = array
            .statistics()
            .compute_stat(Stat::Min)?
            .ok_or_else(|| vortex_err!("Min stat not found"))?;

        let encoded = match_each_integer_ptype!(array.ptype(), |T| {
            compress_primitive::<T>(array, T::try_from(&min)?)?.into_array()
        });
        FoRArray::try_new(encoded, min)
    }
}

#[allow(clippy::cast_possible_truncation)]
fn compress_primitive<T: NativePType + WrappingSub + PrimInt>(
    parray: PrimitiveArray,
    min: T,
) -> VortexResult<PrimitiveArray> {
    // Set null values to the min value, ensuring that decompress into a value in the primitive
    // range (and stop them wrapping around)
    parray.map_each_with_validity::<T, _, _>(|(v, bool)| {
        if bool {
            v.wrapping_sub(&min)
        } else {
            T::zero()
        }
    })
}

/// FoR unpacking strategy that applies a reference value during unpacking
struct FoRStrategy<T> {
    reference: T,
}

impl<T: PhysicalPType<Physical = T> + FoR> UnpackStrategy<T> for FoRStrategy<T> {
    #[inline(always)]
    unsafe fn unpack_chunk(
        &self,
        bit_width: usize,
        chunk: &[T::Physical],
        dst: &mut [T::Physical],
    ) {
        // SAFETY: Caller ensures chunk and dst have correct sizes
        unsafe {
            FoR::unchecked_unfor_pack(bit_width, chunk, self.reference, dst);
        }
    }
}

pub fn decompress(array: &FoRArray) -> PrimitiveArray {
    let ptype = array.ptype();

    // try to do fused unpack
    if array.dtype().is_unsigned_int()
        && let Some(bp) = array.encoded().as_opt::<BitPackedVTable>()
    {
        return match_each_unsigned_integer_ptype!(array.ptype(), |T| {
            fused_decompress::<T>(array, bp)
        });
    }

    // TODO(ngates): do we need this to be into_encoded() somehow?
    let encoded = array.encoded().to_primitive();
    let validity = encoded.validity().clone();

    match_each_integer_ptype!(ptype, |T| {
        let min = array
            .reference_scalar()
            .as_primitive()
            .typed_value::<T>()
            .vortex_expect("reference must be non-null");
        if min == 0 {
            encoded
        } else {
            PrimitiveArray::new(
                decompress_primitive(encoded.into_buffer_mut::<T>(), min),
                validity,
            )
        }
    })
}

fn fused_decompress<T: PhysicalPType<Physical = T> + UnsignedPType + FoR + WrappingAdd>(
    for_: &FoRArray,
    bp: &BitPackedArray,
) -> PrimitiveArray {
    let ref_ = for_
        .reference
        .as_primitive()
        .as_::<T>()
        .vortex_expect("cannot be null");

    let strategy = FoRStrategy { reference: ref_ };

    // Create UnpackedChunks with FoR strategy
    let mut unpacked = UnpackedChunks::new_with_strategy(
        strategy,
        bp.packed().clone(),
        bp.bit_width() as usize,
        bp.offset() as usize,
        bp.len(),
    );

    let mut builder = PrimitiveBuilder::<T>::with_capacity(for_.dtype().nullability(), bp.len());
    let mut uninit_range = builder.uninit_range(bp.len());

    // Decode all chunks (initial, full, and trailer) in one call
    unpacked.decode_into(&mut uninit_range);

    unsafe {
        // Append a dense null Mask.
        uninit_range.append_mask(bp.validity_mask());
    }

    if let Some(patches) = bp.patches() {
        bitpack_decompress::apply_patches_fn(&mut uninit_range, patches, |v| v.wrapping_add(&ref_));
    };

    unsafe {
        uninit_range.finish();
    }

    builder.finish_into_primitive()
}

fn decompress_primitive<T: NativePType + WrappingAdd + PrimInt>(
    values: BufferMut<T>,
    min: T,
) -> Buffer<T> {
    values
        .map_each_in_place(move |v| v.wrapping_add(&min))
        .freeze()
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::stats::StatsProvider;
    use vortex_array::validity::Validity;
    use vortex_array::{ToCanonical, assert_arrays_eq};
    use vortex_buffer::buffer;
    use vortex_dtype::PType;
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn test_compress_round_trip_small() {
        let array = PrimitiveArray::new((1i32..10).collect::<Buffer<_>>(), Validity::NonNullable);
        let compressed = FoRArray::encode(array.clone()).unwrap();
        assert_eq!(i32::try_from(compressed.reference_scalar()).unwrap(), 1);

        let decompressed = compressed.to_primitive();
        assert_arrays_eq!(decompressed, array);
    }

    #[test]
    fn test_compress() {
        // Create a range offset by a million
        let array = PrimitiveArray::new(
            (0u32..10_000).map(|v| v + 1_000_000).collect::<Buffer<_>>(),
            Validity::NonNullable,
        );
        let compressed = FoRArray::encode(array).unwrap();
        assert_eq!(
            u32::try_from(compressed.reference_scalar()).unwrap(),
            1_000_000u32
        );
    }

    #[test]
    fn test_zeros() {
        let array = PrimitiveArray::new(buffer![0i32; 100], Validity::NonNullable);
        assert_eq!(array.statistics().len(), 0);

        let dtype = array.dtype().clone();
        let compressed = FoRArray::encode(array).unwrap();
        assert_eq!(compressed.dtype(), &dtype);
        assert!(compressed.dtype().is_signed_int());
        assert!(compressed.encoded().dtype().is_signed_int());

        let constant = compressed.encoded().as_constant().unwrap();
        assert_eq!(constant, Scalar::from(0i32));
    }

    #[test]
    fn test_decompress() {
        // Create a range offset by a million
        let array = PrimitiveArray::from_iter((0u32..100_000).step_by(1024).map(|v| v + 1_000_000));
        let compressed = FoRArray::encode(array.clone()).unwrap();
        let decompressed = compressed.to_primitive();
        assert_arrays_eq!(decompressed, array);
    }

    #[test]
    fn test_decompress_fused() {
        // Create a range offset by a million
        let expect = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7 + 10));
        let array = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7));
        let bp = BitPackedArray::encode(array.as_ref(), 3).unwrap();
        let compressed = FoRArray::try_new(bp.into_array(), 10u32.into()).unwrap();
        let decompressed = compressed.to_primitive();
        assert_arrays_eq!(decompressed, expect);
    }

    #[test]
    fn test_decompress_fused_patches() {
        // Create a range offset by a million
        let expect = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7 + 10));
        let array = PrimitiveArray::from_iter((0u32..1024).map(|x| x % 7));
        let bp = BitPackedArray::encode(array.as_ref(), 2).unwrap();
        let compressed = FoRArray::try_new(bp.clone().into_array(), 10u32.into()).unwrap();
        let decompressed = fused_decompress::<u32>(&compressed, &bp);
        assert_arrays_eq!(decompressed, expect);
    }

    #[test]
    fn test_overflow() {
        let array = PrimitiveArray::from_iter(i8::MIN..=i8::MAX);
        let compressed = FoRArray::encode(array.clone()).unwrap();
        assert_eq!(
            i8::MIN,
            compressed
                .reference_scalar()
                .as_primitive()
                .typed_value::<i8>()
                .unwrap()
        );

        let encoded = compressed
            .encoded()
            .to_primitive()
            .reinterpret_cast(PType::U8);
        let unsigned: Vec<u8> = (0..=u8::MAX).collect_vec();
        let expected_unsigned = PrimitiveArray::from_iter(unsigned);
        assert_arrays_eq!(encoded, expected_unsigned);

        let decompressed = compressed.to_primitive();
        array
            .as_slice::<i8>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                assert_eq!(*v, i8::try_from(compressed.scalar_at(i).as_ref()).unwrap());
            });
        assert_arrays_eq!(decompressed, array);
    }
}
