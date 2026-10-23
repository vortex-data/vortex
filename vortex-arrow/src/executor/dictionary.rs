// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::ArrowPrimitiveType;
use arrow_array::DictionaryArray;
use arrow_array::PrimitiveArray;
use arrow_array::cast::AsArray;
use arrow_array::new_null_array;
use arrow_array::types::*;
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use arrow_schema::TimeUnit;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::matcher::Matcher;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

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

    // Otherwise, we should try and build a dictionary.
    let array = execute_arrow_naive(array, Some(values_type), exporter, ctx)?;
    if let Some(dict) = primitive_to_dict(&array, codes_type) {
        return Ok(dict);
    }
    // Arrow hides this functionality inside the cast module!
    arrow_cast::cast(
        &array,
        &DataType::Dictionary(Box::new(codes_type.clone()), Box::new(values_type.clone())),
    )
    .map_err(VortexError::from)
}

/// Dictionary-encodes a primitive Arrow array with codes of `codes_type`.
///
/// Arrow's cast hashes every value with SipHash. This hashes the values' bits with FxHash instead,
/// which dominates exporting large nested columns that the reader requests as dictionaries.
/// Returns `None` for unsupported types and when the distinct values overflow the codes type, so
/// the caller can fall back to Arrow's cast.
fn primitive_to_dict(array: &ArrowArrayRef, codes_type: &DataType) -> Option<ArrowArrayRef> {
    match codes_type {
        DataType::Int8 => primitive_to_dict_with::<Int8Type>(array),
        DataType::Int16 => primitive_to_dict_with::<Int16Type>(array),
        DataType::Int32 => primitive_to_dict_with::<Int32Type>(array),
        DataType::Int64 => primitive_to_dict_with::<Int64Type>(array),
        DataType::UInt8 => primitive_to_dict_with::<UInt8Type>(array),
        DataType::UInt16 => primitive_to_dict_with::<UInt16Type>(array),
        DataType::UInt32 => primitive_to_dict_with::<UInt32Type>(array),
        DataType::UInt64 => primitive_to_dict_with::<UInt64Type>(array),
        _ => None,
    }
}

fn primitive_to_dict_with<K: ArrowDictionaryKeyType>(
    array: &ArrowArrayRef,
) -> Option<ArrowArrayRef> {
    match array.data_type() {
        DataType::Int8 => encode_primitive::<K, Int8Type, _>(array, |v| v),
        DataType::Int16 => encode_primitive::<K, Int16Type, _>(array, |v| v),
        DataType::Int32 => encode_primitive::<K, Int32Type, _>(array, |v| v),
        DataType::Int64 => encode_primitive::<K, Int64Type, _>(array, |v| v),
        DataType::UInt8 => encode_primitive::<K, UInt8Type, _>(array, |v| v),
        DataType::UInt16 => encode_primitive::<K, UInt16Type, _>(array, |v| v),
        DataType::UInt32 => encode_primitive::<K, UInt32Type, _>(array, |v| v),
        DataType::UInt64 => encode_primitive::<K, UInt64Type, _>(array, |v| v),
        DataType::Float16 => encode_primitive::<K, Float16Type, _>(array, |v| v.to_bits()),
        DataType::Float32 => encode_primitive::<K, Float32Type, _>(array, |v| v.to_bits()),
        DataType::Float64 => encode_primitive::<K, Float64Type, _>(array, |v| v.to_bits()),
        DataType::Date32 => encode_primitive::<K, Date32Type, _>(array, |v| v),
        DataType::Date64 => encode_primitive::<K, Date64Type, _>(array, |v| v),
        DataType::Timestamp(TimeUnit::Second, _) => {
            encode_primitive::<K, TimestampSecondType, _>(array, |v| v)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            encode_primitive::<K, TimestampMillisecondType, _>(array, |v| v)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            encode_primitive::<K, TimestampMicrosecondType, _>(array, |v| v)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            encode_primitive::<K, TimestampNanosecondType, _>(array, |v| v)
        }
        _ => None,
    }
}

/// Assigns codes in order of first appearance, keyed by `key` of each valid value. Null values
/// get null codes.
fn encode_primitive<K, T, B>(
    array: &ArrowArrayRef,
    key: impl Fn(T::Native) -> B,
) -> Option<ArrowArrayRef>
where
    K: ArrowDictionaryKeyType,
    T: ArrowPrimitiveType,
    B: Hash + Eq,
{
    let array = array.as_primitive::<T>();
    let mut lookup: FxHashMap<B, K::Native> = FxHashMap::default();
    let mut values = Vec::new();
    let mut codes = Vec::with_capacity(array.len());
    for (idx, &value) in array.values().iter().enumerate() {
        if array.is_null(idx) {
            codes.push(K::Native::default());
            continue;
        }
        let code = match lookup.entry(key(value)) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let code = K::Native::from_usize(values.len())?;
                values.push(value);
                *entry.insert(code)
            }
        };
        codes.push(code);
    }
    let codes = PrimitiveArray::<K>::new(codes.into(), array.nulls().cloned());
    let values =
        PrimitiveArray::<T>::new(values.into(), None).with_data_type(array.data_type().clone());
    // SAFETY: every valid code indexes a value pushed above.
    Some(Arc::new(unsafe {
        DictionaryArray::new_unchecked(codes, Arc::new(values))
    }))
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
        DataType::Int8 => Arc::new(PrimitiveArray::<Int8Type>::from_value(0, len)),
        DataType::Int16 => Arc::new(PrimitiveArray::<Int16Type>::from_value(0, len)),
        DataType::Int32 => Arc::new(PrimitiveArray::<Int32Type>::from_value(0, len)),
        DataType::Int64 => Arc::new(PrimitiveArray::<Int64Type>::from_value(0, len)),
        DataType::UInt8 => Arc::new(PrimitiveArray::<UInt8Type>::from_value(0, len)),
        DataType::UInt16 => Arc::new(PrimitiveArray::<UInt16Type>::from_value(0, len)),
        DataType::UInt32 => Arc::new(PrimitiveArray::<UInt32Type>::from_value(0, len)),
        DataType::UInt64 => Arc::new(PrimitiveArray::<UInt64Type>::from_value(0, len)),
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

    use arrow_array::DictionaryArray as ArrowDictArray;
    use arrow_array::Float64Array;
    use arrow_array::Int16Array;
    use arrow_array::Int64Array;
    use arrow_array::StringArray;
    use arrow_array::UInt32Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int16Type;
    use arrow_array::types::UInt8Type;
    use arrow_array::types::UInt32Type;
    use arrow_schema::DataType;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::Nullable;
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
    #[case::primitive_fallback(
        PrimitiveArray::from_option_iter([Some(7i64), None, Some(-3), Some(7), Some(-3)]).into_array(),
        dict_type(DataType::UInt32, DataType::Int64),
        Arc::new(ArrowDictArray::<UInt32Type>::new(
            UInt32Array::from(vec![Some(0), None, Some(1), Some(0), Some(1)]),
            Arc::new(Int64Array::from(vec![7, -3])),
        )) as arrow_array::ArrayRef,
    )]
    #[case::float_fallback_by_bits(
        buffer![0.0f64, -0.0, f64::NAN, 0.0, f64::NAN].into_array(),
        dict_type(DataType::Int16, DataType::Float64),
        Arc::new(ArrowDictArray::<Int16Type>::new(
            Int16Array::from(vec![0, 1, 2, 0, 2]),
            Arc::new(Float64Array::from(vec![0.0, -0.0, f64::NAN])),
        )) as arrow_array::ArrayRef,
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
    fn primitive_fallback_fills_codes_type() -> VortexResult<()> {
        let fits = PrimitiveArray::from_iter(0..=u8::MAX).into_array();
        let actual = execute(fits, &dict_type(DataType::UInt8, DataType::UInt8))?;
        assert_eq!(actual.as_dictionary::<UInt8Type>().values().len(), 256);
        Ok(())
    }

    #[test]
    fn primitive_fallback_overflowing_codes_type_errors() {
        let overflows = PrimitiveArray::from_iter(0..=256u16).into_array();
        assert!(execute(overflows, &dict_type(DataType::UInt8, DataType::UInt16)).is_err());
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
        let flat = arrow_cast::cast(&actual, &DataType::Utf8)?;

        let expected = StringArray::from(vec![Some("a"), None, Some("a")]);
        assert_eq!(flat.as_ref(), &expected as &dyn arrow_array::Array);
        Ok(())
    }
}
