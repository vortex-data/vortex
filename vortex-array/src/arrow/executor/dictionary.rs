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
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::dict::DictArrayParts;
use crate::arrow::ArrowArrayExecutor;

pub(super) fn to_arrow_dictionary(
    array: ArrayRef,
    codes_type: &DataType,
    values_type: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let array = match array.try_into::<Dict>() {
        Ok(dict) => return dict_to_dict(dict, codes_type, values_type, ctx),
        Err(array) => array,
    };
    let array = match array.try_into::<Constant>() {
        Ok(constant) => return constant_to_dict(constant, codes_type, values_type, ctx),
        Err(array) => array,
    };

    // Otherwise, we should try and build a dictionary.
    // Arrow hides this functionality inside the cast module!
    let array = array.execute_arrow(Some(values_type), ctx)?;
    arrow_cast::cast(
        &array,
        &DataType::Dictionary(Box::new(codes_type.clone()), Box::new(values_type.clone())),
    )
    .map_err(VortexError::from)
}

/// Convert a constant array to a dictionary with a single entry.
fn constant_to_dict(
    array: ConstantArray,
    codes_type: &DataType,
    values_type: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();
    let scalar = array.scalar();
    if scalar.is_null() {
        let dict_type =
            DataType::Dictionary(Box::new(codes_type.clone()), Box::new(values_type.clone()));
        return Ok(new_null_array(&dict_type, len));
    }

    let values = ConstantArray::new(scalar.clone(), 1)
        .into_array()
        .execute_arrow(Some(values_type), ctx)?;
    let codes = zeroed_codes_array(codes_type, len)?;
    make_dict_array(codes_type, codes, values)
}

/// Convert a Vortex dictionary array to an Arrow dictionary array.
fn dict_to_dict(
    array: DictArray,
    codes_type: &DataType,
    values_type: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let DictArrayParts { codes, values, .. } = array.into_data().into_parts();
    let codes = codes.execute_arrow(Some(codes_type), ctx)?;
    let values = values.execute_arrow(Some(values_type), ctx)?;
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

    use arrow_array::DictionaryArray as ArrowDictArray;
    use arrow_array::types::UInt8Type;
    use arrow_array::types::UInt32Type;
    use arrow_schema::DataType;
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrow::ArrowArrayExecutor;
    use crate::arrow::executor::dictionary::ConstantArray;
    use crate::arrow::executor::dictionary::DictArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability::Nullable;
    use crate::executor::VortexSessionExecute;
    use crate::scalar::Scalar;

    fn dict_type(codes: DataType, values: DataType) -> DataType {
        DataType::Dictionary(Box::new(codes), Box::new(values))
    }

    fn execute(array: crate::ArrayRef, dt: &DataType) -> VortexResult<arrow_array::ArrayRef> {
        array.execute_arrow(Some(dt), &mut LEGACY_SESSION.create_execution_ctx())
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
        DictArray::try_new(
            buffer![0u8, 1, 0].into_array(),
            VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
        ).unwrap().into_array(),
        dict_type(DataType::UInt8, DataType::Utf8),
        Arc::new(vec![Some("a"), Some("b"), Some("a")].into_iter().collect::<ArrowDictArray<UInt8Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::dict_with_null_codes(
        DictArray::try_new(
            PrimitiveArray::from_option_iter(vec![Some(0u8), None, Some(1)]).into_array(),
            VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
        ).unwrap().into_array(),
        dict_type(DataType::UInt8, DataType::Utf8),
        Arc::new(vec![Some("a"), None, Some("b")].into_iter().collect::<ArrowDictArray<UInt8Type>>()) as arrow_array::ArrayRef,
    )]
    #[case::varbinview_fallback(
        [Some("a"), None, Some("a"), Some("b"), Some("a")].into_iter().collect::<VarBinViewArray>().into_array(),
        dict_type(DataType::UInt8, DataType::Utf8),
        Arc::new(vec![Some("a"), None, Some("a"), Some("b"), Some("a")].into_iter().collect::<ArrowDictArray<UInt8Type>>()) as arrow_array::ArrayRef,
    )]
    fn to_arrow_dictionary(
        #[case] input: crate::ArrayRef,
        #[case] target_type: DataType,
        #[case] expected: arrow_array::ArrayRef,
    ) -> VortexResult<()> {
        let actual = execute(input, &target_type)?;
        assert_eq!(expected.as_ref(), actual.as_ref());
        Ok(())
    }
}
