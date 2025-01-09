use num_traits::{PrimInt, WrappingAdd, WrappingSub};
use vortex_array::array::{ConstantArray, PrimitiveArray, SparseArray};
use vortex_array::stats::{trailing_zeros, ArrayStatistics, Stat};
use vortex_array::validity::LogicalValidity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{
    match_each_integer_ptype, match_each_unsigned_integer_ptype, DType, NativePType, Nullability,
};
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::FoRArray;

pub fn for_compress(array: PrimitiveArray) -> VortexResult<FoRArray> {
    let shift = trailing_zeros(array.as_ref());
    let min = array
        .statistics()
        .compute(Stat::Min)
        .ok_or_else(|| vortex_err!("Min stat not found"))?;

    let nullability = array.dtype().nullability();
    let encoded = match_each_integer_ptype!(array.ptype(), |$T| {
        if shift == <$T>::PTYPE.bit_width() as u8 {
            assert_eq!(min, Scalar::zero::<$T>(array.dtype().nullability()));
            encoded_zero::<$T>(array.validity().to_logical(array.len()), nullability)
                .vortex_expect("Failed to encode all zeroes")
        } else {
            let unsigned_ptype = array.ptype().to_unsigned();
            compress_primitive::<$T>(array, shift, $T::try_from(&min)?)
                .reinterpret_cast(unsigned_ptype)
                .into_array()
        }
    });
    FoRArray::try_new(encoded, min, shift)
}

fn encoded_zero<T: NativePType>(
    logical_validity: LogicalValidity,
    nullability: Nullability,
) -> VortexResult<ArrayData> {
    if nullability == Nullability::NonNullable
        && !matches!(logical_validity, LogicalValidity::AllValid(_))
    {
        vortex_bail!("Must have LogicalValidity::AllValid with non-nullable DType")
    }

    let encoded_ptype = T::PTYPE.to_unsigned();
    let zero =
        match_each_unsigned_integer_ptype!(encoded_ptype, |$T| Scalar::zero::<$T>(nullability));

    Ok(match logical_validity {
        LogicalValidity::AllValid(len) => ConstantArray::new(zero, len).into_array(),
        LogicalValidity::AllInvalid(len) => ConstantArray::new(
            Scalar::null(DType::Primitive(encoded_ptype, nullability)),
            len,
        )
        .into_array(),
        LogicalValidity::Array(a) => {
            let len = a.len();
            let valid_indices = a
                .into_bool()?
                .boolean_buffer()
                .set_indices()
                .map(|i| i as u64)
                .collect::<Buffer<u64>>()
                .into_array();
            let valid_len = valid_indices.len();
            SparseArray::try_new(
                valid_indices,
                ConstantArray::new(zero, valid_len).into_array(),
                len,
                Scalar::null(DType::Primitive(encoded_ptype, Nullability::Nullable)),
            )?
            .into_array()
        }
    })
}

#[allow(clippy::cast_possible_truncation)]
fn compress_primitive<T: NativePType + WrappingSub + PrimInt>(
    parray: PrimitiveArray,
    shift: u8,
    min: T,
) -> PrimitiveArray {
    assert!(shift < T::PTYPE.bit_width() as u8);
    if shift > 0 {
        parray.map_each::<T, _, _>(|v| v.wrapping_sub(&min) >> (shift as usize))
    } else {
        parray.map_each::<T, _, _>(|v| v.wrapping_sub(&min))
    }
}

pub fn decompress(array: FoRArray) -> VortexResult<PrimitiveArray> {
    let shift = array.shift() as usize;
    let ptype = array.ptype();

    // TODO(ngates): do we need this to be into_encoded() somehow?
    let encoded = array.encoded().into_primitive()?.reinterpret_cast(ptype);
    let validity = encoded.validity();

    Ok(match_each_integer_ptype!(ptype, |$T| {
        if shift == <$T>::PTYPE.bit_width() {
            encoded
        } else {
            let min = array.reference_scalar()
                .as_primitive()
                .typed_value::<$T>()
                .ok_or_else(|| vortex_err!("expected reference to be non-null"))?;
            if min == 0 && shift == 0 {
                encoded
            } else {
                PrimitiveArray::new(
                    decompress_primitive(encoded.into_buffer_mut::<$T>(), min, shift),
                    validity,
                )
            }
        }
    }))
}

fn decompress_primitive<T: NativePType + WrappingAdd + PrimInt>(
    values: BufferMut<T>,
    min: T,
    shift: usize,
) -> Buffer<T> {
    if shift > 0 {
        if min == T::zero() {
            values.map_each(move |v| v << shift).freeze()
        } else {
            values
                .map_each(move |v| (v << shift).wrapping_add(&min))
                .freeze()
        }
    } else {
        values.map_each(move |v| v.wrapping_add(&min)).freeze()
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::compute::scalar_at;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayVariant;
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;

    use super::*;

    #[test]
    fn test_compress() {
        // Create a range offset by a million
        let array = PrimitiveArray::new(
            (0u32..10_000).map(|v| v + 1_000_000).collect::<Buffer<_>>(),
            Validity::NonNullable,
        );
        let compressed = for_compress(array).unwrap();
        assert_eq!(
            u32::try_from(compressed.reference_scalar()).unwrap(),
            1_000_000u32
        );
    }

    #[test]
    fn test_zeros() {
        let array = PrimitiveArray::new(buffer![0i32; 100], Validity::NonNullable);
        assert!(array.statistics().to_set().into_iter().next().is_none());

        let compressed = for_compress(array.clone()).unwrap();
        assert_eq!(compressed.dtype(), array.dtype());
        assert!(compressed.dtype().is_signed_int());
        assert!(compressed.encoded().dtype().is_unsigned_int());

        let constant = compressed.encoded().as_constant().unwrap();
        assert_eq!(constant, Scalar::from(0u32));
    }

    #[test]
    fn test_nullable_zeros() {
        let array =
            PrimitiveArray::from_option_iter([Some(0i32), None].into_iter().cycle().take(100));
        assert!(array.statistics().to_set().into_iter().next().is_none());

        let compressed = for_compress(array.clone()).unwrap();
        assert_eq!(compressed.dtype(), array.dtype());
        assert!(compressed.dtype().is_signed_int());
        assert_eq!(
            scalar_at(&compressed, 0).unwrap(),
            Scalar::primitive(0i32, Nullability::Nullable)
        );
        assert_eq!(
            scalar_at(&compressed, 1).unwrap(),
            Scalar::null(array.dtype().clone())
        );

        let sparse = SparseArray::try_from(compressed.encoded()).unwrap();
        assert!(sparse.dtype().is_unsigned_int());
        assert!(sparse.statistics().to_set().into_iter().next().is_none());
        assert_eq!(sparse.fill_scalar(), Scalar::null(sparse.dtype().clone()));
        assert_eq!(
            scalar_at(&sparse, 0).unwrap(),
            Scalar::primitive(0u32, Nullability::Nullable)
        );
        assert_eq!(
            scalar_at(&sparse, 1).unwrap(),
            Scalar::null(sparse.dtype().clone())
        );
    }

    #[test]
    fn test_decompress() {
        // Create a range offset by a million
        let array = PrimitiveArray::from_iter((0u32..100_000).step_by(1024).map(|v| v + 1_000_000));
        let compressed = for_compress(array.clone()).unwrap();
        assert!(compressed.shift() > 0);
        let decompressed = compressed.into_primitive().unwrap();
        assert_eq!(decompressed.as_slice::<u32>(), array.as_slice::<u32>());
    }

    #[test]
    fn test_overflow() {
        let array = PrimitiveArray::from_iter(i8::MIN..=i8::MAX);
        let compressed = for_compress(array.clone()).unwrap();
        assert_eq!(
            i8::MIN,
            compressed
                .reference_scalar()
                .as_primitive()
                .typed_value::<i8>()
                .unwrap()
        );

        let encoded = compressed.encoded().into_primitive().unwrap();
        let encoded_bytes: &[u8] = encoded.as_slice::<u8>();
        let unsigned: Vec<u8> = (0..=u8::MAX).collect_vec();
        assert_eq!(encoded_bytes, unsigned.as_slice());

        let decompressed = compressed.as_ref().clone().into_primitive().unwrap();
        assert_eq!(decompressed.as_slice::<i8>(), array.as_slice::<i8>());
        array
            .as_slice::<i8>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                assert_eq!(
                    *v,
                    i8::try_from(scalar_at(&compressed, i).unwrap().as_ref()).unwrap()
                );
            });
    }
}
