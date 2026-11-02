// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::DictionaryArray;
use arrow_array::PrimitiveArray;
use arrow_array::cast::AsArray;
use arrow_array::new_null_array;
use arrow_array::types::*;
use arrow_schema::DataType;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray as VortexPrimitiveArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builders::dict::DictConstraints;
use vortex_array::builders::dict::dict_encoder_in;
use vortex_array::dtype::DType;
use vortex_array::matcher::Matcher;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrowExporter;
use crate::dtype::ptype_from_arrow;
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

    if let DType::Primitive(ptype, _) = array.dtype()
        && ptype_from_arrow(values_type).ok() == Some(*ptype)
    {
        let dict = encode_primitive(array, ctx)?;
        return dict_to_dict(dict, codes_type, values_type, exporter, ctx);
    }

    // Otherwise, we should try and build a dictionary.
    // Arrow hides this functionality inside the cast module!
    let array = execute_arrow_naive(array, Some(values_type), exporter, ctx)?;
    arrow_cast::cast(
        &array,
        &DataType::Dictionary(Box::new(codes_type.clone()), Box::new(values_type.clone())),
    )
    .map_err(VortexError::from)
}

/// Dictionary-encode a primitive array stored without a dictionary. `arrow_cast` builds the same
/// values, but its builder hashes with SipHash and looks up every new value twice.
fn encode_primitive(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DictArray> {
    // The encoder only accepts canonical arrays.
    let array = array.execute::<VortexPrimitiveArray>(ctx)?.into_array();
    let len = array.len();
    // No input has more distinct values than rows, so capping the dictionary at the input length
    // never stops the encoder early and picks the narrowest codes that hold every value.
    let constraints = DictConstraints {
        max_bytes: usize::MAX,
        max_len: len,
    };
    let mut encoder = dict_encoder_in(&array, &constraints, ctx.allocator().clone());
    // Sized for all-distinct input like Arrow's builder: growing the table rehashes every entry.
    encoder.reserve(len);
    let codes = encoder.encode(&array, ctx)?;
    // SAFETY: the encoder only emits codes for values it returns from `reset`.
    Ok(unsafe { DictArray::new_unchecked(codes.into_array(), encoder.reset()) })
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
    use arrow_array::StringArray;
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
    fn to_arrow_dictionary(
        #[case] input: vortex_array::ArrayRef,
        #[case] target_type: DataType,
        #[case] expected: arrow_array::ArrayRef,
    ) -> VortexResult<()> {
        let actual = execute(input, &target_type)?;
        assert_eq!(expected.as_ref(), actual.as_ref());
        Ok(())
    }

    /// Primitive arrays stored without a dictionary export a dictionary that decodes to the same rows.
    #[rstest]
    #[case::u64_with_nulls(
        PrimitiveArray::from_option_iter([Some(7u64), None, Some(9), Some(7), None, Some(1)]).into_array(),
        DataType::UInt32,
        DataType::UInt64,
    )]
    #[case::signed_keys(buffer![-1i32, 5, -1, 5, 5, 2].into_array(), DataType::Int8, DataType::Int32)]
    #[case::float_bits(
        buffer![0.0f64, -0.0, f64::NAN, 0.0, f64::NAN].into_array(),
        DataType::UInt16,
        DataType::Float64,
    )]
    #[case::all_null(
        PrimitiveArray::from_option_iter([None::<u32>, None]).into_array(),
        DataType::UInt32,
        DataType::UInt32,
    )]
    fn primitive_dict_decodes_to_same_rows(
        #[case] array: vortex_array::ArrayRef,
        #[case] codes_type: DataType,
        #[case] values_type: DataType,
    ) -> VortexResult<()> {
        let target_type = dict_type(codes_type, values_type.clone());
        let expected = execute(array.clone(), &values_type)?;
        let actual = execute(array, &target_type)?;
        assert_eq!(actual.data_type(), &target_type);
        // Nulls are dictionary values rather than null keys, so compare the decoded rows.
        let decoded = arrow_cast::cast(&actual, &values_type)?;
        assert_eq!(decoded.as_ref(), expected.as_ref());
        Ok(())
    }

    #[test]
    fn primitive_overflowing_key_type_errors() {
        let array = PrimitiveArray::from_iter(0u32..300).into_array();
        assert!(execute(array, &dict_type(DataType::UInt8, DataType::UInt32)).is_err());
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
