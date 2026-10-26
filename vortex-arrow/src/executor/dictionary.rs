// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::DictionaryArray;
use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
use arrow_array::cast::AsArray;
use arrow_array::new_null_array;
use arrow_array::types::*;
use arrow_schema::DataType;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::decimal::DecimalArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::builders::dict::dict_encode;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::matcher::Matcher;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrowExporter;
use crate::executor::execute_arrow_naive;

/// Matches the encodings [`to_arrow_dictionary`] requires for export.
struct ArrowDictExportable;

impl Matcher for ArrowDictExportable {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<Dict>() || array.is::<Constant>()).then_some(array)
    }
}

pub(super) fn to_arrow_dictionary(
    array: ArrayRef,
    codes_type: &DataType,
    values_type: &DataType,
    exporter: &ArrowExporter<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let array = array.execute_until::<ArrowDictExportable>(ctx)?;

    let array = match array.try_downcast::<Dict>() {
        Ok(dict) => return dict_to_dict(dict, codes_type, values_type, exporter, ctx),
        Err(array) => array,
    };
    let array = match array.try_downcast::<Constant>() {
        Ok(constant) => return constant_to_dict(constant, codes_type, values_type, exporter, ctx),
        Err(array) => array,
    };

    // Otherwise, dictionary-encode the array ourselves.
    let dict = encode_dict(array, ctx)?;
    dict_to_dict(dict, codes_type, values_type, exporter, ctx)
}

/// Dictionary-encode an array that is not already dictionary encoded.
///
/// Supports the value types Arrow can dictionary-encode: primitives, decimals, strings and binary,
/// and extension types stored as one of these. Nulls are carried by the codes, matching how Arrow
/// builds dictionaries.
fn encode_dict(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DictArray> {
    match array.execute::<Canonical>(ctx)? {
        canonical @ (Canonical::Primitive(_) | Canonical::VarBinView(_)) => {
            let array = canonical.into_array();
            let validity = array.is_not_null()?;
            // `dict_encode` keeps nulls as a dictionary value, so move them onto the codes.
            let dict = dict_encode(&array, ctx)?;
            DictArray::try_new(dict.codes().clone().mask(validity)?, dict.values().clone())
        }
        Canonical::Decimal(decimal) => {
            match_each_decimal_value_type!(decimal.values_type(), |T| {
                encode_decimal_dict::<T>(&decimal, ctx)
            })
        }
        Canonical::Extension(ext) => {
            let storage_dict = encode_dict(ext.storage_array().clone(), ctx)?;
            let values =
                ExtensionArray::try_new(ext.ext_dtype().clone(), storage_dict.values().clone())?;
            DictArray::try_new(storage_dict.codes().clone(), values.into_array())
        }
        canonical => vortex_bail!(
            "Cannot dictionary-encode Vortex array with dtype {} for Arrow export",
            canonical.dtype()
        ),
    }
}

/// Dictionary-encode a decimal array whose values are stored as `T`.
fn encode_decimal_dict<T: NativeDecimalType + Hash>(
    array: &DecimalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DictArray> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    let values = array.buffer::<T>();

    let mut lookup = HashMap::<T, u64>::default();
    let mut unique = BufferMut::<T>::empty();
    let mut codes = BufferMut::<u64>::with_capacity(values.len());
    for (&value, is_valid) in values.iter().zip(validity.to_bit_buffer().iter()) {
        let code = if is_valid {
            *lookup.entry(value).or_insert_with(|| {
                unique.push(value);
                (unique.len() - 1) as u64
            })
        } else {
            0
        };
        codes.push(code);
    }

    let codes = PrimitiveArray::new(
        codes.freeze(),
        Validity::from_mask(validity, array.dtype().nullability()),
    );
    let values = DecimalArray::new(
        unique.freeze(),
        array.decimal_dtype(),
        Validity::NonNullable,
    );
    DictArray::try_new(codes.into_array(), values.into_array())
}

/// Convert a constant array to a dictionary with a single entry.
fn constant_to_dict(
    array: ConstantArray,
    codes_type: &DataType,
    values_type: &DataType,
    exporter: &ArrowExporter<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();
    let scalar = array.scalar();
    if scalar.is_null() {
        let dict_type =
            DataType::Dictionary(Box::new(codes_type.clone()), Box::new(values_type.clone()));
        return Ok(new_null_array(&dict_type, len));
    }

    let values = execute_arrow_naive(
        ConstantArray::new(scalar.clone(), 1).into_array(),
        Some(values_type),
        exporter,
        ctx,
    )?;
    let codes = zeroed_codes_array(codes_type, len)?;
    make_dict_array(codes_type, codes, values)
}

/// Convert a Vortex dictionary array to an Arrow dictionary array.
fn dict_to_dict(
    array: DictArray,
    codes_type: &DataType,
    values_type: &DataType,
    exporter: &ArrowExporter<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let codes = execute_arrow_naive(array.codes().clone(), Some(codes_type), exporter, ctx)?;
    let values = execute_arrow_naive(array.values().clone(), Some(values_type), exporter, ctx)?;
    make_dict_array(codes_type, codes, values)
}

/// Construct a zeroed Arrow primitive array directly.
fn zeroed_codes_array(codes_type: &DataType, len: usize) -> VortexResult<ArrowArrayRef> {
    Ok(match codes_type {
        DataType::Int8 => Arc::new(ArrowPrimitiveArray::<Int8Type>::from_value(0, len)),
        DataType::Int16 => Arc::new(ArrowPrimitiveArray::<Int16Type>::from_value(0, len)),
        DataType::Int32 => Arc::new(ArrowPrimitiveArray::<Int32Type>::from_value(0, len)),
        DataType::Int64 => Arc::new(ArrowPrimitiveArray::<Int64Type>::from_value(0, len)),
        DataType::UInt8 => Arc::new(ArrowPrimitiveArray::<UInt8Type>::from_value(0, len)),
        DataType::UInt16 => Arc::new(ArrowPrimitiveArray::<UInt16Type>::from_value(0, len)),
        DataType::UInt32 => Arc::new(ArrowPrimitiveArray::<UInt32Type>::from_value(0, len)),
        DataType::UInt64 => Arc::new(ArrowPrimitiveArray::<UInt64Type>::from_value(0, len)),
        _ => vortex_bail!("Unsupported dictionary codes type: {:?}", codes_type),
    })
}

/// Construct an Arrow `DictionaryArray` from pre-built codes and values arrays.
fn make_dict_array(
    codes_type: &DataType,
    codes: ArrowArrayRef,
    values: ArrowArrayRef,
) -> VortexResult<ArrowArrayRef> {
    Ok(match codes_type {
        DataType::Int8 => Arc::new(unsafe {
            DictionaryArray::new_unchecked(codes.as_primitive::<Int8Type>().clone(), values)
        }),
        DataType::Int16 => Arc::new(unsafe {
            DictionaryArray::new_unchecked(codes.as_primitive::<Int16Type>().clone(), values)
        }),
        DataType::Int32 => Arc::new(unsafe {
            DictionaryArray::new_unchecked(codes.as_primitive::<Int32Type>().clone(), values)
        }),
        DataType::Int64 => Arc::new(unsafe {
            DictionaryArray::new_unchecked(codes.as_primitive::<Int64Type>().clone(), values)
        }),
        DataType::UInt8 => Arc::new(unsafe {
            DictionaryArray::new_unchecked(codes.as_primitive::<UInt8Type>().clone(), values)
        }),
        DataType::UInt16 => Arc::new(unsafe {
            DictionaryArray::new_unchecked(codes.as_primitive::<UInt16Type>().clone(), values)
        }),
        DataType::UInt32 => Arc::new(unsafe {
            DictionaryArray::new_unchecked(codes.as_primitive::<UInt32Type>().clone(), values)
        }),
        DataType::UInt64 => Arc::new(unsafe {
            DictionaryArray::new_unchecked(codes.as_primitive::<UInt64Type>().clone(), values)
        }),
        _ => vortex_bail!("Unsupported dictionary codes type: {:?}", codes_type),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::Decimal128Array;
    use arrow_array::DictionaryArray as ArrowDictArray;
    use arrow_array::Int32Array;
    use arrow_array::TimestampSecondArray;
    use arrow_array::UInt8Array;
    use arrow_array::types::Int32Type;
    use arrow_array::types::UInt8Type;
    use arrow_array::types::UInt32Type;
    use arrow_schema::DataType;
    use arrow_schema::TimeUnit as ArrowTimeUnit;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability::Nullable;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::mask::Mask;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrowArrayExecutor;
    use crate::executor::dictionary::ConstantArray;
    use crate::executor::dictionary::DictArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn dict_type(codes: DataType, values: DataType) -> DataType {
        DataType::Dictionary(Box::new(codes), Box::new(values))
    }

    fn execute(
        array: vortex_array::ArrayRef,
        dt: &DataType,
    ) -> VortexResult<arrow_array::ArrayRef> {
        array.execute_arrow(Some(dt), &mut SESSION.create_execution_ctx())
    }

    fn dict_basic_input() -> vortex_array::ArrayRef {
        DictArray::try_new(
            buffer![0u8, 1, 0].into_array(),
            VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
        )
        .expect("valid dictionary input")
        .into_array()
    }

    fn dict_with_null_codes_input() -> vortex_array::ArrayRef {
        DictArray::try_new(
            PrimitiveArray::from_option_iter(vec![Some(0u8), None, Some(1)]).into_array(),
            VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
        )
        .expect("valid dictionary input with null codes")
        .into_array()
    }

    #[rstest]
    #[case::constant_null(
        ConstantArray::new(Scalar::null(DType::Utf8(Nullable)), 4).into_array(),
        dict_type(DataType::UInt32, DataType::Utf8),
        Arc::new(vec![None::<&str>, None, None, None].into_iter().collect::<ArrowDictArray<UInt32Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::constant_non_null(
        ConstantArray::new(Scalar::from("hello"), 5).into_array(),
        dict_type(DataType::UInt32, DataType::Utf8),
        Arc::new(vec![Some("hello"); 5].into_iter().collect::<ArrowDictArray<UInt32Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::dict_basic(
        dict_basic_input(),
        dict_type(DataType::UInt8, DataType::Utf8),
        Arc::new(vec![Some("a"), Some("b"), Some("a")].into_iter().collect::<ArrowDictArray<UInt8Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::dict_with_null_codes(
        dict_with_null_codes_input(),
        dict_type(DataType::UInt8, DataType::Utf8),
        Arc::new(vec![Some("a"), None, Some("b")].into_iter().collect::<ArrowDictArray<UInt8Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::varbinview_fallback(
        [Some("a"), None, Some("a"), Some("b"), Some("a")].into_iter().collect::<VarBinViewArray>().into_array(),
        dict_type(DataType::UInt8, DataType::Utf8),
        Arc::new(vec![Some("a"), None, Some("a"), Some("b"), Some("a")].into_iter().collect::<ArrowDictArray<UInt8Type>>()) as arrow_array::ArrayRef,
    )]
    fn to_arrow_dictionary(
        #[case] input: vortex_array::ArrayRef,
        #[case] target_type: DataType,
        #[case] expected: arrow_array::ArrayRef,
    ) -> VortexResult<()> {
        let actual = execute(input, &target_type)?;
        assert_eq!(expected.as_ref(), actual.as_ref());
        Ok(())
    }

    #[test]
    fn nullable_primitive_fallback() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(7i32), None, Some(9), Some(7)]).into_array();
        let actual = execute(array, &dict_type(DataType::UInt8, DataType::Int32))?;

        let expected: ArrowDictArray<UInt8Type> = ArrowDictArray::try_new(
            UInt8Array::from(vec![Some(0), None, Some(1), Some(0)]),
            Arc::new(Int32Array::from(vec![7, 9])),
        )?;
        assert_eq!(&expected as &dyn arrow_array::Array, actual.as_ref());
        Ok(())
    }

    #[test]
    fn nullable_decimal_fallback() -> VortexResult<()> {
        let array = DecimalArray::from_option_iter(
            [Some(10i128), None, Some(20), Some(10)],
            DecimalDType::new(20, 2),
        )
        .into_array();
        let values_type = DataType::Decimal128(20, 2);
        let actual = execute(array, &dict_type(DataType::UInt8, values_type))?;

        let values = Decimal128Array::from(vec![10, 20]).with_precision_and_scale(20, 2)?;
        let expected: ArrowDictArray<UInt8Type> = ArrowDictArray::try_new(
            UInt8Array::from(vec![Some(0), None, Some(1), Some(0)]),
            Arc::new(values),
        )?;
        assert_eq!(&expected as &dyn arrow_array::Array, actual.as_ref());
        Ok(())
    }

    #[test]
    fn timestamp_fallback() -> VortexResult<()> {
        let storage = PrimitiveArray::from_option_iter([Some(5i64), None, Some(5), Some(7)]);
        let array = TemporalArray::new_timestamp(storage.into_array(), TimeUnit::Seconds, None);
        let values_type = DataType::Timestamp(ArrowTimeUnit::Second, None);
        let actual = execute(array.into(), &dict_type(DataType::Int32, values_type))?;

        let expected: ArrowDictArray<Int32Type> = ArrowDictArray::try_new(
            Int32Array::from(vec![Some(0), None, Some(0), Some(1)]),
            Arc::new(TimestampSecondArray::from(vec![5, 7])),
        )?;
        assert_eq!(&expected as &dyn arrow_array::Array, actual.as_ref());
        Ok(())
    }

    #[test]
    fn codes_overflow_errors() {
        let array = PrimitiveArray::from_iter(0..300i32).into_array();
        assert!(execute(array, &dict_type(DataType::UInt8, DataType::Int32)).is_err());
    }

    #[test]
    fn unsupported_fallback_errors() {
        let array = BoolArray::from_iter([true, false, true]).into_array();
        assert!(execute(array, &dict_type(DataType::UInt8, DataType::Boolean)).is_err());
    }

    #[test]
    fn mask_wrapped_dict_exports() -> VortexResult<()> {
        // Dictionary behind a lazy `mask` scalar-fn — the shape a scan produces when a row
        // mask is applied to a dict-encoded column.
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0].into_array(),
            VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
        )?;
        let mask = BoolArray::from_iter([true, false, true]);
        let masked = Mask::try_new(dict.into_array(), mask.into_array())?.into_array();

        let actual = execute(masked, &dict_type(DataType::UInt8, DataType::Utf8))?;
        let expected: ArrowDictArray<UInt8Type> =
            vec![Some("a"), None, Some("a")].into_iter().collect();
        assert_eq!(&expected as &dyn arrow_array::Array, actual.as_ref());
        Ok(())
    }
}
