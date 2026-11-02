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
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray as VortexPrimitiveArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::matcher::Matcher;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;

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

/// Matches [`ArrowDictExportable`] encodings and chunked arrays, whose chunks may each be one.
struct ArrowDictExportableOrChunked;

impl Matcher for ArrowDictExportableOrChunked {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (ArrowDictExportable::matches(array) || array.is::<Chunked>()).then_some(array)
    }
}

pub(super) fn to_arrow_dictionary(
    array: ArrayRef,
    codes_type: &DataType,
    values_type: &DataType,
    exporter: &ArrowExporter<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let array = array.execute_until::<ArrowDictExportableOrChunked>(ctx)?;
    let array = match array.try_downcast::<Chunked>() {
        Ok(chunked) => {
            if let Some(dict) = chunked_to_dict(&chunked, codes_type, values_type, exporter, ctx)? {
                return Ok(dict);
            }
            chunked.into_array()
        }
        Err(array) => array,
    };
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
    // Arrow hides this functionality inside the cast module!
    let array = execute_arrow_naive(array, Some(values_type), exporter, ctx)?;
    arrow_cast::cast(
        &array,
        &DataType::Dictionary(Box::new(codes_type.clone()), Box::new(values_type.clone())),
    )
    .map_err(VortexError::from)
}

/// Joins the dictionaries of a chunked array into one Arrow dictionary without hashing.
///
/// The chunks' values are appended and each chunk's codes are shifted past the values before
/// it; a constant chunk contributes one value. Values are not deduplicated, which Arrow
/// dictionaries permit. Returns `None` unless every chunk executes to a dictionary or a
/// constant and the joined values fit in `codes_type`.
fn chunked_to_dict(
    chunked: &ChunkedArray,
    codes_type: &DataType,
    values_type: &DataType,
    exporter: &ArrowExporter<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrowArrayRef>> {
    let dtype = chunked.dtype().clone();
    let mut codes = BufferMut::<u64>::with_capacity(chunked.len());
    let mut validity = BitBufferMut::with_capacity(chunked.len());
    let mut values = Vec::with_capacity(chunked.nchunks());
    let mut offset = 0u64;

    for chunk in chunked.non_empty_chunks() {
        let chunk = chunk.clone().execute_until::<ArrowDictExportable>(ctx)?;
        let chunk_values = match chunk.try_downcast::<Dict>() {
            Ok(dict) => {
                let chunk_codes = dict.codes().clone().execute::<VortexPrimitiveArray>(ctx)?;
                let mask = chunk_codes
                    .as_ref()
                    .validity()?
                    .execute_mask(chunk_codes.len(), ctx)?;
                // Null codes may hold any value, so they point at the first joined value.
                match_each_integer_ptype!(chunk_codes.ptype(), |P| {
                    let chunk_codes = chunk_codes.as_slice::<P>();
                    match mask.bit_buffer() {
                        AllOr::All => codes.extend(
                            chunk_codes
                                .iter()
                                .map(|&code| AsPrimitive::<u64>::as_(code) + offset),
                        ),
                        AllOr::None => codes.push_n(0, chunk_codes.len()),
                        AllOr::Some(bits) => codes.extend(chunk_codes.iter().zip(bits.iter()).map(
                            |(&code, valid)| {
                                if valid {
                                    AsPrimitive::<u64>::as_(code) + offset
                                } else {
                                    0
                                }
                            },
                        )),
                    }
                });
                validity.append_buffer(&mask.into_bit_buffer());
                dict.values().cast(dtype.clone())?
            }
            Err(chunk) => {
                let Ok(constant) = chunk.try_downcast::<Constant>() else {
                    return Ok(None);
                };
                codes.push_n(offset, constant.len());
                validity.append_n(!constant.scalar().is_null(), constant.len());
                ConstantArray::new(constant.scalar().clone(), 1).into_array()
            }
        };
        offset += chunk_values.len() as u64;
        values.push(chunk_values);
    }

    if offset == 0 || offset - 1 > max_code(codes_type)? {
        return Ok(None);
    }

    let codes = VortexPrimitiveArray::new(
        codes.freeze(),
        Validity::from_bit_buffer(validity.freeze(), Nullability::Nullable),
    );
    let codes = execute_arrow_naive(codes.into_array(), Some(codes_type), exporter, ctx)?;
    let values = ChunkedArray::try_new(values, dtype)?.into_array();
    let values = execute_arrow_naive(values, Some(values_type), exporter, ctx)?;
    make_dict_array(codes_type, codes, values).map(Some)
}

/// The largest code an Arrow dictionary with `codes_type` keys can hold.
fn max_code(codes_type: &DataType) -> VortexResult<u64> {
    Ok(match codes_type {
        DataType::Int8 => i8::MAX as u64,
        DataType::Int16 => i16::MAX as u64,
        DataType::Int32 => i32::MAX as u64,
        DataType::Int64 => i64::MAX as u64,
        DataType::UInt8 => u8::MAX as u64,
        DataType::UInt16 => u16::MAX as u64,
        DataType::UInt32 => u32::MAX as u64,
        DataType::UInt64 => u64::MAX,
        _ => vortex_bail!("Unsupported dictionary codes type: {:?}", codes_type),
    })
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
    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt8Type;
    use arrow_array::types::UInt32Type;
    use arrow_schema::DataType;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ChunkedArray;
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

    #[test]
    fn chunked_dicts_join_without_hashing() -> VortexResult<()> {
        let dict = |codes: PrimitiveArray, values: [&str; 2]| -> VortexResult<_> {
            let values = values.into_iter().map(Some).collect::<VarBinViewArray>();
            Ok(DictArray::try_new(codes.into_array(), values.into_array())?.into_array())
        };
        let chunks = vec![
            dict(PrimitiveArray::from_iter([0u8, 1, 0]), ["a", "b"])?,
            ConstantArray::new(Scalar::null(DType::Utf8(Nullable)), 2).into_array(),
            dict(
                PrimitiveArray::from_option_iter([Some(1u8), None]),
                ["c", "a"],
            )?,
        ];
        let chunked = ChunkedArray::try_new(chunks, DType::Utf8(Nullable))?.into_array();

        let actual = execute(chunked, &dict_type(DataType::UInt32, DataType::Utf8))?;

        // Two values per dictionary plus one for the constant: appended, not deduplicated.
        assert_eq!(actual.as_any_dictionary().values().len(), 5);
        let flat = arrow_cast::cast(&actual, &DataType::Utf8)?;
        let expected = StringArray::from(vec![
            Some("a"),
            Some("b"),
            Some("a"),
            None,
            None,
            Some("a"),
            None,
        ]);
        assert_eq!(flat.as_ref(), &expected as &dyn arrow_array::Array);
        Ok(())
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
