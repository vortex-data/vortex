use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{PrimitiveArray, VarBinArray, VarBinViewArray};
use vortex_array::ArrayDType;
use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::VortexExpect as _;

use crate::primitive_builder::{NullablePrimitiveDictionaryBuilder, PrimitiveDictionaryBuilder};
use crate::varbin_builder::{NullableVarBinDictionaryBuilder, VarBinDictionaryBuilder};
use crate::{NullableVarBinViewDictionaryBuilder, VarBinViewDictionaryBuilder};

pub fn dict_encode_primitive(array: &PrimitiveArray) -> (PrimitiveArray, PrimitiveArray) {
    match_each_native_ptype!(array.ptype(), |$P| {
        dict_encode_typed_primitive::<$P>(array)
    })
}

/// Dictionary encode primitive array with given PType.
/// Null values in the original array are encoded in the dictionary.
pub fn dict_encode_typed_primitive<T: NativePType>(
    array: &PrimitiveArray,
) -> (PrimitiveArray, PrimitiveArray) {
    if array.dtype().is_nullable() {
        let mut builder = NullablePrimitiveDictionaryBuilder::<T>::with_capacity(array.len());
        array
            .with_iterator(|iter| iter.map(|v| v.cloned()).for_each(|v| builder.append(v)))
            .vortex_expect("Failed to dictionary encode primitive array");
        builder
            .into_parts()
            .vortex_expect("Failed to extract dict encoded values out of builder")
    } else {
        let mut builder = PrimitiveDictionaryBuilder::with_capacity(array.len());
        for val in array.maybe_null_slice::<T>() {
            builder.append_value(*val);
        }
        let (codes, values) = builder.into_parts();
        (PrimitiveArray::from(codes), PrimitiveArray::from(values))
    }
}

macro_rules! dict_encode_var {
    ($arr:ident, $builder:ty, $null_builder:ty) => {
        if $arr.dtype().is_nullable() {
            let mut builder = <$null_builder>::with_capacity($arr.len());
            $arr.with_iterator(|iter| iter.for_each(|v| builder.append(v)))
                .vortex_expect("Failed to dictionary encode primitive array");
            builder
                .into_parts($arr.dtype().clone())
                .vortex_expect("Failed to extract dict encoded values out of builder")
        } else {
            let mut builder = <$builder>::with_capacity($arr.len());
            $arr.with_iterator(|iter| {
                iter.map(|v| v.vortex_expect("non nullable value"))
                    .for_each(|v| builder.append_value(v))
            })
            .vortex_expect("Failed to dictionary encode primitive array");
            builder
                .into_parts($arr.dtype().clone())
                .vortex_expect("Unable to extract parts out of array")
        }
    };
}

/// Dictionary encode varbin array. Specializes for primitive byte arrays to avoid double copying
pub fn dict_encode_varbin(array: &VarBinArray) -> (PrimitiveArray, VarBinArray) {
    dict_encode_var!(
        array,
        VarBinDictionaryBuilder,
        NullableVarBinDictionaryBuilder
    )
}

/// Dictionary encode a VarbinViewArray.
pub fn dict_encode_varbinview(array: &VarBinViewArray) -> (PrimitiveArray, VarBinViewArray) {
    dict_encode_var!(
        array,
        VarBinViewDictionaryBuilder,
        NullableVarBinViewDictionaryBuilder
    )
}

#[cfg(test)]
mod test {
    use std::str;

    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::array::{PrimitiveArray, VarBinArray};
    use vortex_array::compute::unary::scalar_at;
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::{DType, PType};
    use vortex_scalar::Scalar;

    use crate::compress::{dict_encode_typed_primitive, dict_encode_varbin};

    #[test]
    fn encode_primitive() {
        let arr = PrimitiveArray::from(vec![1, 1, 3, 3, 3]);
        let (codes, values) = dict_encode_typed_primitive::<i32>(&arr);
        assert_eq!(codes.maybe_null_slice::<u64>(), &[0, 0, 1, 1, 1]);
        assert_eq!(values.maybe_null_slice::<i32>(), &[1, 3]);
    }

    #[test]
    fn encode_primitive_nulls() {
        let arr = PrimitiveArray::from_nullable_vec(vec![
            Some(1),
            Some(1),
            None,
            Some(3),
            Some(3),
            None,
            Some(3),
            None,
        ]);
        let (codes, values) = dict_encode_typed_primitive::<i32>(&arr);
        assert_eq!(codes.maybe_null_slice::<u64>(), &[1, 1, 0, 2, 2, 0, 2, 0]);
        assert_eq!(
            scalar_at(&values, 0).unwrap(),
            Scalar::null(DType::Primitive(PType::I32, Nullable))
        );
        assert_eq!(
            scalar_at(&values, 1).unwrap(),
            Scalar::primitive(1, Nullable)
        );
        assert_eq!(
            scalar_at(&values, 2).unwrap(),
            Scalar::primitive(3, Nullable)
        );
    }

    #[test]
    fn encode_varbin() {
        let arr = VarBinArray::from(vec!["hello", "world", "hello", "again", "world"]);
        let (codes, values) = dict_encode_varbin(&arr);
        assert_eq!(codes.maybe_null_slice::<u64>(), &[0, 1, 0, 2, 1]);
        values
            .with_iterator(|iter| {
                assert_eq!(
                    iter.flatten()
                        .map(|b| unsafe { str::from_utf8_unchecked(b) })
                        .collect::<Vec<_>>(),
                    vec!["hello", "world", "again"]
                );
            })
            .unwrap();
    }

    #[test]
    fn encode_varbin_nulls() {
        let arr: VarBinArray = vec![
            Some("hello"),
            None,
            Some("world"),
            Some("hello"),
            None,
            Some("again"),
            Some("world"),
            None,
        ]
        .into_iter()
        .collect();
        let (codes, values) = dict_encode_varbin(&arr);
        assert_eq!(codes.maybe_null_slice::<u64>(), &[1, 0, 2, 1, 0, 3, 2, 0]);
        assert_eq!(str::from_utf8(&values.bytes_at(0).unwrap()).unwrap(), "");
        values
            .with_iterator(|iter| {
                assert_eq!(
                    iter.map(|b| b.map(|v| unsafe { str::from_utf8_unchecked(v) }))
                        .collect::<Vec<_>>(),
                    vec![None, Some("hello"), Some("world"), Some("again")]
                );
            })
            .unwrap();
    }

    #[test]
    fn repeated_values() {
        let arr = VarBinArray::from(vec!["a", "a", "b", "b", "a", "b", "a", "b"]);
        let (codes, values) = dict_encode_varbin(&arr);
        values
            .with_iterator(|iter| {
                assert_eq!(
                    iter.flatten()
                        .map(|b| unsafe { str::from_utf8_unchecked(b) })
                        .collect::<Vec<_>>(),
                    vec!["a", "b"]
                );
            })
            .unwrap();
        assert_eq!(
            codes.maybe_null_slice::<u64>(),
            &[0u64, 0, 1, 1, 0, 1, 0, 1]
        );
    }
}
