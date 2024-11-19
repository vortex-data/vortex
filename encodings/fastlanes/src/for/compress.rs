use itertools::Itertools;
use num_traits::{PrimInt, WrappingAdd, WrappingSub};
use vortex_array::array::{ConstantArray, PrimitiveArray, SparseArray};
use vortex_array::stats::{trailing_zeros, ArrayStatistics, Stat};
use vortex_array::validity::LogicalValidity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::FoRArray;

pub fn for_compress(array: &PrimitiveArray) -> VortexResult<ArrayData> {
    let shift = trailing_zeros(array.as_ref());
    let min = array
        .statistics()
        .compute(Stat::Min)
        .ok_or_else(|| vortex_err!("Min stat not found"))?;

    Ok(match_each_integer_ptype!(array.ptype(), |$T| {
        if shift == <$T>::PTYPE.bit_width() as u8 {
            match array.validity().to_logical(array.len()) {
                LogicalValidity::AllValid(l) => {
                    ConstantArray::new(Scalar::zero::<$T>(array.dtype().nullability()), l).into_array()
                },
                LogicalValidity::AllInvalid(l) => {
                    ConstantArray::new(Scalar::null(array.dtype().clone()), l).into_array()
                }
                LogicalValidity::Array(a) => {
                    let valid_indices = PrimitiveArray::from(
                        a.into_bool()?
                            .boolean_buffer()
                            .set_indices()
                            .map(|i| i as u64)
                            .collect::<Vec<_>>(),
                    )
                    .into_array();
                    let valid_len = valid_indices.len();
                    SparseArray::try_new(
                        valid_indices,
                        ConstantArray::new(Scalar::zero::<$T>(array.dtype().nullability()), valid_len)
                            .into_array(),
                        array.len(),
                        ScalarValue::Null,
                    )?
                    .into_array()
                }
            }
        } else {
            FoRArray::try_new(
                compress_primitive::<$T>(&array, shift, $T::try_from(&min)?)
                    .reinterpret_cast(array.ptype().to_unsigned())
                    .into_array(),
                min,
                shift,
            )?
            .into_array()
        }
    }))
}

fn compress_primitive<T: NativePType + WrappingSub + PrimInt>(
    parray: &PrimitiveArray,
    shift: u8,
    min: T,
) -> PrimitiveArray {
    assert!(shift < T::PTYPE.bit_width() as u8);
    let values = if shift > 0 {
        parray
            .maybe_null_slice::<T>()
            .iter()
            .map(|&v| v.wrapping_sub(&min))
            .map(|v| v >> shift as usize)
            .collect_vec()
    } else {
        parray
            .maybe_null_slice::<T>()
            .iter()
            .map(|&v| v.wrapping_sub(&min))
            .collect_vec()
    };

    PrimitiveArray::from_vec(values, parray.validity())
}

pub fn decompress(array: FoRArray) -> VortexResult<PrimitiveArray> {
    let shift = array.shift() as usize;
    let ptype = array.ptype();
    let encoded = array.encoded().into_primitive()?.reinterpret_cast(ptype);
    let validity = encoded.validity();
    Ok(match_each_integer_ptype!(ptype, |$T| {
        let min: $T = array.reference().try_into()?;
        PrimitiveArray::from_vec(
            decompress_primitive(encoded.into_maybe_null_slice::<$T>(), min, shift),
            validity,
        )
    }))
}

fn decompress_primitive<T: NativePType + WrappingAdd + PrimInt>(
    values: Vec<T>,
    min: T,
    shift: usize,
) -> Vec<T> {
    if shift > 0 {
        values
            .into_iter()
            .map(|v| v << shift)
            .map(|v| v.wrapping_add(&min))
            .collect_vec()
    } else {
        values
            .into_iter()
            .map(|v| v.wrapping_add(&min))
            .collect_vec()
    }
}

#[cfg(test)]
mod test {
    use vortex_array::compute::unary::ScalarAtFn;
    use vortex_array::IntoArrayVariant;
    use vortex_dtype::Nullability;

    use super::*;

    #[test]
    fn test_compress() {
        // Create a range offset by a million
        let array = PrimitiveArray::from((0u32..10_000).map(|v| v + 1_000_000).collect_vec());
        let compressed = FoRArray::try_from(for_compress(&array).unwrap()).unwrap();

        assert_eq!(u32::try_from(compressed.reference()).unwrap(), 1_000_000u32);
    }

    #[test]
    fn test_zeros() {
        let array = PrimitiveArray::from(vec![0i32; 10_000]);
        assert!(array.statistics().to_set().into_iter().next().is_none());

        let compressed = for_compress(&array).unwrap();
        let constant = ConstantArray::try_from(compressed).unwrap();
        assert_eq!(constant.scalar_value(), &ScalarValue::from(0i32));
    }

    #[test]
    fn test_nullable_zeros() {
        let array = PrimitiveArray::from_nullable_vec(
            vec![Some(0i32), None]
                .into_iter()
                .cycle()
                .take(10_000)
                .collect_vec(),
        );
        assert!(array.statistics().to_set().into_iter().next().is_none());

        let compressed = for_compress(&array).unwrap();
        let sparse = SparseArray::try_from(compressed).unwrap();
        assert!(sparse.statistics().to_set().into_iter().next().is_none());
        assert_eq!(sparse.fill_value(), &ScalarValue::Null);
        assert_eq!(
            sparse.scalar_at(0).unwrap(),
            Scalar::primitive(0i32, Nullability::Nullable)
        );
        assert_eq!(
            sparse.scalar_at(1).unwrap(),
            Scalar::null(sparse.dtype().clone())
        );
    }

    #[test]
    fn test_decompress() {
        // Create a range offset by a million
        let array = PrimitiveArray::from(
            (0u32..100_000)
                .step_by(1024)
                .map(|v| v + 1_000_000)
                .collect_vec(),
        );
        let compressed = for_compress(&array).unwrap();
        let for_arr = FoRArray::try_from(compressed.clone()).unwrap();
        assert!(for_arr.shift() > 0);
        let decompressed = compressed.into_primitive().unwrap();
        assert_eq!(
            decompressed.maybe_null_slice::<u32>(),
            array.maybe_null_slice::<u32>()
        );
    }

    #[test]
    fn test_overflow() {
        let array = PrimitiveArray::from((i8::MIN..=i8::MAX).collect_vec());
        let compressed = FoRArray::try_from(for_compress(&array).unwrap()).unwrap();
        assert_eq!(i8::MIN, i8::try_from(compressed.reference()).unwrap());

        let encoded = compressed.encoded().into_primitive().unwrap();
        let encoded_bytes: &[u8] = encoded.maybe_null_slice::<u8>();
        let unsigned: Vec<u8> = (0..=u8::MAX).collect_vec();
        assert_eq!(encoded_bytes, unsigned.as_slice());

        let decompressed = compressed.as_ref().clone().into_primitive().unwrap();
        assert_eq!(
            decompressed.maybe_null_slice::<i8>(),
            array.maybe_null_slice::<i8>()
        );
        array
            .maybe_null_slice::<i8>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| {
                assert_eq!(
                    *v,
                    i8::try_from(compressed.scalar_at(i).unwrap().as_ref()).unwrap()
                );
            });
    }
}
