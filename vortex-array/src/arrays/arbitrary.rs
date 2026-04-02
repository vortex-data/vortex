// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::Arc;

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::builders::ArrayBuilder;
use crate::builders::DecimalBuilder;
use crate::builders::FixedSizeListBuilder;
use crate::builders::ListViewBuilder;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_decimal_value_type;
use crate::scalar::Scalar;
use crate::scalar::arbitrary::random_scalar;
use crate::validity::Validity;

/// A wrapper type to implement `Arbitrary` for `ArrayRef`.
#[derive(Clone, Debug)]
pub struct ArbitraryArray(pub ArrayRef);

impl<'a> Arbitrary<'a> for ArbitraryArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let dtype = u.arbitrary()?;
        Self::arbitrary_with(u, None, &dtype)
    }
}

impl ArbitraryArray {
    pub fn arbitrary_with(u: &mut Unstructured, len: Option<usize>, dtype: &DType) -> Result<Self> {
        random_array(u, dtype, len).map(ArbitraryArray)
    }
}

fn split_number_into_parts(n: usize, parts: usize) -> Vec<usize> {
    let reminder = n % parts;
    let division = (n - reminder) / parts;
    iter::repeat_n(division, parts - reminder)
        .chain(iter::repeat_n(division + 1, reminder))
        .collect()
}

/// Creates a random array with a random number of chunks.
fn random_array(u: &mut Unstructured, dtype: &DType, len: Option<usize>) -> Result<ArrayRef> {
    let num_chunks = u.int_in_range(1..=3)?;
    let chunk_lens = len.map(|l| split_number_into_parts(l, num_chunks));
    let mut chunks = (0..num_chunks)
        .map(|i| {
            let chunk_len = chunk_lens.as_ref().map(|c| c[i]);
            random_array_chunk(u, dtype, chunk_len)
        })
        .collect::<Result<Vec<_>>>()?;

    if chunks.len() == 1 {
        Ok(chunks.remove(0))
    } else {
        let dtype = chunks[0].dtype().clone();
        Ok(ChunkedArray::try_new(chunks, dtype)
            .vortex_expect("operation should succeed in arbitrary impl")
            .into_array())
    }
}

/// Creates a random array chunk.
fn random_array_chunk(
    u: &mut Unstructured<'_>,
    dtype: &DType,
    chunk_len: Option<usize>,
) -> Result<ArrayRef> {
    match dtype {
        DType::Null => Ok(NullArray::new(
            chunk_len
                .map(Ok)
                .unwrap_or_else(|| u.int_in_range(0..=100))?,
        )
        .into_array()),
        DType::Bool(n) => random_bool(u, *n, chunk_len),
        DType::Primitive(ptype, n) => match ptype {
            PType::U8 => random_primitive::<u8>(u, *n, chunk_len),
            PType::U16 => random_primitive::<u16>(u, *n, chunk_len),
            PType::U32 => random_primitive::<u32>(u, *n, chunk_len),
            PType::U64 => random_primitive::<u64>(u, *n, chunk_len),
            PType::I8 => random_primitive::<i8>(u, *n, chunk_len),
            PType::I16 => random_primitive::<i16>(u, *n, chunk_len),
            PType::I32 => random_primitive::<i32>(u, *n, chunk_len),
            PType::I64 => random_primitive::<i64>(u, *n, chunk_len),
            PType::F16 => Ok(random_primitive::<u16>(u, *n, chunk_len)?
                .to_primitive()
                .reinterpret_cast(PType::F16)
                .into_array()),
            PType::F32 => random_primitive::<f32>(u, *n, chunk_len),
            PType::F64 => random_primitive::<f64>(u, *n, chunk_len),
        },
        d @ DType::Decimal(decimal, n) => {
            let elem_len = chunk_len.unwrap_or(u.int_in_range(0..=20)?);
            match_each_decimal_value_type!(DecimalType::smallest_decimal_value_type(decimal), |D| {
                let mut builder = DecimalBuilder::new::<D>(*decimal, *n);
                for _i in 0..elem_len {
                    let random_decimal = random_scalar(u, d)?;
                    builder.append_scalar(&random_decimal).vortex_expect(
                        "was somehow unable to append a decimal to a decimal builder",
                    );
                }
                Ok(builder.finish())
            })
        }
        DType::Utf8(n) => random_string(u, *n, chunk_len),
        DType::Binary(n) => random_bytes(u, *n, chunk_len),
        DType::Struct(sdt, n) => {
            let first_array = sdt
                .fields()
                .next()
                .map(|d| random_array(u, &d, chunk_len))
                .transpose()?;
            let resolved_len = first_array
                .as_ref()
                .map(|a| a.len())
                .or(chunk_len)
                .map(Ok)
                .unwrap_or_else(|| u.int_in_range(0..=100))?;
            let children = first_array
                .into_iter()
                .map(Ok)
                .chain(
                    sdt.fields()
                        .skip(1)
                        .map(|d| random_array(u, &d, Some(resolved_len))),
                )
                .collect::<Result<Vec<_>>>()?;
            Ok(StructArray::try_new(
                sdt.names().clone(),
                children,
                resolved_len,
                random_validity(u, *n, resolved_len)?,
            )
            .vortex_expect("operation should succeed in arbitrary impl")
            .into_array())
        }
        DType::List(elem_dtype, null) => random_list(u, elem_dtype, *null, chunk_len),
        DType::FixedSizeList(elem_dtype, list_size, null) => {
            random_fixed_size_list(u, elem_dtype, *list_size, *null, chunk_len)
        }
        DType::Extension(..) => {
            unimplemented!("Extension arrays are not implemented")
        }
        DType::Variant(_) => {
            unimplemented!("Variant arrays are not implemented")
        }
    }
}

/// Creates a random fixed-size list array.
///
/// If the `chunk_len` is specified, the length of the array will be equal to the chunk length.
fn random_fixed_size_list(
    u: &mut Unstructured,
    elem_dtype: &Arc<DType>,
    list_size: u32,
    null: Nullability,
    chunk_len: Option<usize>,
) -> Result<ArrayRef> {
    let array_length = chunk_len.unwrap_or(u.int_in_range(0..=20)?);

    let mut builder =
        FixedSizeListBuilder::with_capacity(elem_dtype.clone(), list_size, null, array_length);

    for _ in 0..array_length {
        if null == Nullability::Nullable && u.arbitrary::<bool>()? {
            builder.append_null();
        } else {
            builder
                .append_value(random_list_scalar(u, elem_dtype, list_size, null)?.as_list())
                .vortex_expect("can append value");
        }
    }

    Ok(builder.finish())
}

/// Creates a random list array.
///
/// If the `chunk_len` is specified, the length of the array will be equal to the chunk length.
fn random_list(
    u: &mut Unstructured,
    elem_dtype: &Arc<DType>,
    null: Nullability,
    chunk_len: Option<usize>,
) -> Result<ArrayRef> {
    match u.int_in_range(0..=5)? {
        0 => random_list_with_offset_type::<i16>(u, elem_dtype, null, chunk_len),
        1 => random_list_with_offset_type::<i32>(u, elem_dtype, null, chunk_len),
        2 => random_list_with_offset_type::<i64>(u, elem_dtype, null, chunk_len),
        3 => random_list_with_offset_type::<u16>(u, elem_dtype, null, chunk_len),
        4 => random_list_with_offset_type::<u32>(u, elem_dtype, null, chunk_len),
        5 => random_list_with_offset_type::<u64>(u, elem_dtype, null, chunk_len),
        _ => unreachable!("int_in_range returns a value in the above range"),
    }
}

/// Creates a random list array with the given [`IntegerPType`] for the internal offsets child.
///
/// If the `chunk_len` is specified, the length of the array will be equal to the chunk length.
fn random_list_with_offset_type<O: IntegerPType>(
    u: &mut Unstructured,
    elem_dtype: &Arc<DType>,
    null: Nullability,
    chunk_len: Option<usize>,
) -> Result<ArrayRef> {
    let array_length = chunk_len.unwrap_or(u.int_in_range(0..=20)?);

    let mut builder = ListViewBuilder::<O, O>::with_capacity(elem_dtype.clone(), null, 20, 10);

    for _ in 0..array_length {
        if null == Nullability::Nullable && u.arbitrary::<bool>()? {
            builder.append_null();
        } else {
            let list_size = u.int_in_range(0..=20)?;
            builder
                .append_value(random_list_scalar(u, elem_dtype, list_size, null)?.as_list())
                .vortex_expect("can append value");
        }
    }

    Ok(builder.finish())
}

/// Creates a random list scalar with the specified list size.
fn random_list_scalar(
    u: &mut Unstructured,
    elem_dtype: &Arc<DType>,
    list_size: u32,
    null: Nullability,
) -> Result<Scalar> {
    let elems = (0..list_size)
        .map(|_| random_scalar(u, elem_dtype))
        .collect::<Result<Vec<_>>>()?;
    Ok(Scalar::list(elem_dtype.clone(), elems, null))
}

fn random_string(
    u: &mut Unstructured,
    nullability: Nullability,
    len: Option<usize>,
) -> Result<ArrayRef> {
    match nullability {
        Nullability::NonNullable => {
            let v = arbitrary_vec_of_len::<String>(u, len)?;
            Ok(match u.int_in_range(0..=1)? {
                0 => VarBinArray::from_vec(v, DType::Utf8(Nullability::NonNullable)).into_array(),
                1 => VarBinViewArray::from_iter_str(v).into_array(),
                _ => unreachable!(),
            })
        }
        Nullability::Nullable => {
            let v = arbitrary_vec_of_len::<Option<String>>(u, len)?;
            Ok(match u.int_in_range(0..=1)? {
                0 => VarBinArray::from_iter(v, DType::Utf8(Nullability::Nullable)).into_array(),
                1 => VarBinViewArray::from_iter_nullable_str(v).into_array(),
                _ => unreachable!(),
            })
        }
    }
}

fn random_bytes(
    u: &mut Unstructured,
    nullability: Nullability,
    len: Option<usize>,
) -> Result<ArrayRef> {
    match nullability {
        Nullability::NonNullable => {
            let v = arbitrary_vec_of_len::<Vec<u8>>(u, len)?;
            Ok(match u.int_in_range(0..=1)? {
                0 => VarBinArray::from_vec(v, DType::Binary(Nullability::NonNullable)).into_array(),
                1 => VarBinViewArray::from_iter_bin(v).into_array(),
                _ => unreachable!(),
            })
        }
        Nullability::Nullable => {
            let v = arbitrary_vec_of_len::<Option<Vec<u8>>>(u, len)?;
            Ok(match u.int_in_range(0..=1)? {
                0 => VarBinArray::from_iter(v, DType::Binary(Nullability::Nullable)).into_array(),
                1 => VarBinViewArray::from_iter_nullable_bin(v).into_array(),
                _ => unreachable!(),
            })
        }
    }
}

fn random_primitive<'a, T: Arbitrary<'a> + NativePType>(
    u: &mut Unstructured<'a>,
    nullability: Nullability,
    len: Option<usize>,
) -> Result<ArrayRef> {
    let v = arbitrary_vec_of_len::<T>(u, len)?;
    let validity = random_validity(u, nullability, v.len())?;
    Ok(PrimitiveArray::new(Buffer::copy_from(v), validity).into_array())
}

fn random_bool(
    u: &mut Unstructured,
    nullability: Nullability,
    len: Option<usize>,
) -> Result<ArrayRef> {
    let v = arbitrary_vec_of_len(u, len)?;
    let validity = random_validity(u, nullability, v.len())?;
    Ok(BoolArray::new(BitBuffer::from(v), validity).into_array())
}

pub fn random_validity(
    u: &mut Unstructured,
    nullability: Nullability,
    len: usize,
) -> Result<Validity> {
    match nullability {
        Nullability::NonNullable => Ok(Validity::NonNullable),
        Nullability::Nullable => Ok(match u.int_in_range(0..=2)? {
            0 => Validity::AllValid,
            1 => Validity::AllInvalid,
            2 => Validity::from_iter(arbitrary_vec_of_len::<bool>(u, Some(len))?),
            _ => unreachable!(),
        }),
    }
}

fn arbitrary_vec_of_len<'a, T: Arbitrary<'a>>(
    u: &mut Unstructured<'a>,
    len: Option<usize>,
) -> Result<Vec<T>> {
    len.map(|l| (0..l).map(|_| T::arbitrary(u)).collect::<Result<Vec<_>>>())
        .unwrap_or_else(|| Vec::<T>::arbitrary(u))
}
