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
use arrow_buffer::ScalarBuffer;
use arrow_schema::DataType;
use arrow_schema::TimeUnit;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray as VortexPrimitiveArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::matcher::Matcher;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::BitWidthsView;

use crate::ArrowExporter;
use crate::executor::execute_arrow_naive;

/// Matches the encodings [`to_arrow_dictionary`] requires for export.
struct ArrowDictExportable;

impl Matcher for ArrowDictExportable {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<Dict>() || array.is::<Constant>() || array.is::<BitPacked>()).then_some(array)
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
    if let Some(dict) = bitpacked_to_dict(&array, codes_type, values_type, exporter, ctx)? {
        return Ok(dict);
    }

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

/// The widest packed values [`bitpacked_to_dict`] exports as codes.
const MAX_BITPACKED_CODE_BITS: u8 = 16;

/// Exports a patch-free bit-packed array as a dictionary over every value its bit width can hold.
///
/// Packed values are non-negative and below `2^bit_width`, so the unpacked values are their own
/// codes into the values `0..2^bit_width`. Unpacking straight into the codes type skips decoding
/// to the values type and then dictionary-encoding the result. Returns `None` when the array is not
/// such a bit-packed array, the codes type cannot hold every packed value, or the dictionary would
/// outnumber the array's rows.
fn bitpacked_to_dict(
    array: &ArrayRef,
    codes_type: &DataType,
    values_type: &DataType,
    exporter: &ArrowExporter<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrowArrayRef>> {
    let Some(bitpacked) = array.as_opt::<BitPacked>() else {
        return Ok(None);
    };
    let BitWidthsView::Global(bit_width) = bitpacked.bit_widths() else {
        return Ok(None);
    };
    let Some(codes_ptype) = codes_ptype(codes_type) else {
        return Ok(None);
    };
    let dict_len = 1usize << bit_width;
    if bit_width > MAX_BITPACKED_CODE_BITS
        || usize::from(bit_width) + usize::from(codes_ptype.is_signed_int())
            > codes_ptype.bit_width()
        || !values_type.is_integer()
        || dict_len > array.len()
        || bitpacked.patches().is_some()
    {
        return Ok(None);
    }

    let codes = array
        .clone()
        .cast(DType::Primitive(codes_ptype, array.dtype().nullability()))?;
    let codes = execute_arrow_naive(codes, Some(codes_type), exporter, ctx)?;
    let values = match_each_integer_ptype!(array.dtype().as_ptype(), |T| {
        VortexPrimitiveArray::from_iter((0..dict_len).map(T::usize_as)).into_array()
    });
    let values = execute_arrow_naive(values, Some(values_type), exporter, ctx)?;
    make_dict_array(codes_type, codes, values).map(Some)
}

fn codes_ptype(codes_type: &DataType) -> Option<PType> {
    Some(match codes_type {
        DataType::Int8 => PType::I8,
        DataType::Int16 => PType::I16,
        DataType::Int32 => PType::I32,
        DataType::Int64 => PType::I64,
        DataType::UInt8 => PType::U8,
        DataType::UInt16 => PType::U16,
        DataType::UInt32 => PType::U32,
        DataType::UInt64 => PType::U64,
        _ => return None,
    })
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

/// Valid values sampled to choose how [`encode_primitive`] assigns codes.
const SAMPLE_LEN: usize = 1024;
/// The largest key range [`encode_primitive`] looks up in a dense table.
const MAX_DENSE_RANGE: usize = 1 << 16;

/// Dictionary-encodes `array` keyed by `key` of each valid value. Null values get null codes.
///
/// Keys spanning a small range are looked up in a dense table. When a sample of the values is
/// nearly all distinct, the values themselves become the dictionary with sequential codes, since
/// Arrow dictionaries may hold duplicates and deduplicating would barely shrink them. Otherwise
/// codes are assigned in order of first appearance through a hash map.
fn encode_primitive<K, T, B>(
    array: &ArrowArrayRef,
    key: impl Fn(T::Native) -> B,
) -> Option<ArrowArrayRef>
where
    K: ArrowDictionaryKeyType,
    T: ArrowPrimitiveType,
    B: Hash + Eq + Copy + Into<i128>,
{
    let array = array.as_primitive::<T>();
    let nulls = array.nulls().filter(|nulls| nulls.null_count() > 0);
    let is_valid = |idx: usize| nulls.is_none_or(|nulls| nulls.is_valid(idx));
    let valid_keys = || {
        array
            .values()
            .iter()
            .enumerate()
            .filter(move |(idx, _)| is_valid(*idx))
            .map(|(_, &value)| key(value))
    };

    let (min, max) = valid_keys().fold((i128::MAX, i128::MIN), |(min, max), key| {
        let key = key.into();
        (min.min(key), max.max(key))
    });
    let range = usize::try_from(max.checked_sub(min)?).ok();
    let (codes, values) = match range {
        Some(range) if range < MAX_DENSE_RANGE && range <= array.len().saturating_mul(4) => {
            encode_dense::<K, T, B>(array, &is_valid, &key, min, range)?
        }
        _ => {
            let mut sample = FxHashSet::default();
            let sampled = valid_keys()
                .take(SAMPLE_LEN)
                .inspect(|key| {
                    sample.insert(*key);
                })
                .count();
            if sample.len() * 8 >= sampled * 7 && K::Native::from_usize(array.len()).is_some() {
                let codes = (0..array.len()).map(K::Native::usize_as).collect();
                (codes, array.values().clone())
            } else {
                let distinct = sample.len() * array.len() / sampled.max(1);
                encode_hashed::<K, T, B>(array, &is_valid, &key, distinct)?
            }
        }
    };

    let codes = PrimitiveArray::<K>::new(codes, nulls.cloned());
    let values = PrimitiveArray::<T>::new(values, None).with_data_type(array.data_type().clone());
    // SAFETY: every valid code indexes a value of the dictionary.
    Some(Arc::new(unsafe {
        DictionaryArray::new_unchecked(codes, Arc::new(values))
    }))
}

/// Assigns codes in order of first appearance through a table indexed by `key - min`.
fn encode_dense<K, T, B>(
    array: &PrimitiveArray<T>,
    is_valid: &impl Fn(usize) -> bool,
    key: &impl Fn(T::Native) -> B,
    min: i128,
    range: usize,
) -> Option<(ScalarBuffer<K::Native>, ScalarBuffer<T::Native>)>
where
    K: ArrowDictionaryKeyType,
    T: ArrowPrimitiveType,
    B: Into<i128>,
{
    const UNSEEN: u32 = u32::MAX;
    let mut table = vec![UNSEEN; range + 1];
    let mut values = Vec::new();
    let mut codes = Vec::with_capacity(array.len());
    for (idx, &value) in array.values().iter().enumerate() {
        if !is_valid(idx) {
            codes.push(K::Native::default());
            continue;
        }
        let slot = &mut table[usize::try_from(key(value).into() - min).ok()?];
        if *slot == UNSEEN {
            K::Native::from_usize(values.len())?;
            *slot = u32::try_from(values.len()).ok()?;
            values.push(value);
        }
        codes.push(K::Native::usize_as(*slot as usize));
    }
    Some((codes.into(), values.into()))
}

/// Assigns codes in order of first appearance through a hash map sized for `distinct` keys.
fn encode_hashed<K, T, B>(
    array: &PrimitiveArray<T>,
    is_valid: &impl Fn(usize) -> bool,
    key: &impl Fn(T::Native) -> B,
    distinct: usize,
) -> Option<(ScalarBuffer<K::Native>, ScalarBuffer<T::Native>)>
where
    K: ArrowDictionaryKeyType,
    T: ArrowPrimitiveType,
    B: Hash + Eq,
{
    let mut lookup: FxHashMap<B, K::Native> =
        FxHashMap::with_capacity_and_hasher(distinct, Default::default());
    let mut values = Vec::with_capacity(distinct);
    let mut codes = Vec::with_capacity(array.len());
    for (idx, &value) in array.values().iter().enumerate() {
        if !is_valid(idx) {
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
    Some((codes.into(), values.into()))
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
    use arrow_array::UInt64Array;
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
    use vortex_fastlanes::BitPackedArrayExt;
    use vortex_fastlanes::BitPackedData;
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
    fn primitive_fallback_distinct_values() -> VortexResult<()> {
        let values = (0..2000u64).map(|i| (i % 7 != 3).then(|| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        let input = PrimitiveArray::from_option_iter(values.clone()).into_array();
        let actual = execute(input, &dict_type(DataType::UInt32, DataType::UInt64))?;
        let expected = UInt64Array::from_iter(values);
        let flat = arrow_cast::cast(&actual, &DataType::UInt64)?;
        assert_eq!(flat.as_ref(), &expected as &dyn arrow_array::Array);
        Ok(())
    }

    #[rstest]
    #[case::fits(DataType::UInt32, 11)]
    #[case::signed_codes(DataType::Int16, 11)]
    #[case::codes_narrower_than_bit_width(DataType::UInt16, 20)]
    #[case::wide(DataType::UInt32, 20)]
    fn bitpacked_exports_as_dict(
        #[case] codes_type: DataType,
        #[case] bit_width: u8,
    ) -> VortexResult<()> {
        let max = (1u64 << bit_width) - 1;
        let values: Vec<_> = (0..5000u64)
            .map(|i| (i % 13 != 4).then(|| i64::try_from(i * 7919 % max).unwrap_or(0)))
            .collect();
        let source = PrimitiveArray::from_option_iter(values.clone()).into_array();
        let packed =
            BitPackedData::encode(&source, bit_width, &mut SESSION.create_execution_ctx())?;
        assert!(packed.patches().is_none());

        let actual = execute(packed.into_array(), &dict_type(codes_type, DataType::Int64))?;
        let flat = arrow_cast::cast(&actual, &DataType::Int64)?;
        assert_eq!(flat.as_ref(), &Int64Array::from(values) as &dyn arrow_array::Array);
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
