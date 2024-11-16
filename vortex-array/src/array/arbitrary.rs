use std::iter;

use arbitrary::{Arbitrary, Result, Unstructured};
use arrow_buffer::BooleanBuffer;
use vortex_dtype::{DType, NativePType, Nullability, PType};
use vortex_error::{VortexExpect, VortexUnwrap};

use super::{BoolArray, ChunkedArray, NullArray, PrimitiveArray, StructArray};
use crate::array::{VarBinArray, VarBinViewArray};
use crate::validity::Validity;
use crate::{ArrayDType, ArrayData, IntoArrayData as _, IntoArrayVariant};

impl<'a> Arbitrary<'a> for ArrayData {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let dtype = u.arbitrary()?;
        random_array(u, &dtype, None)
    }
}

fn random_array(u: &mut Unstructured, dtype: &DType, len: Option<usize>) -> Result<ArrayData> {
    let num_chunks = u.int_in_range(1..=3)?;
    let chunk_lens = len.map(|l| split_number_into_parts(l, num_chunks));
    let mut chunks = (0..num_chunks)
        .map(|i| {
            let chunk_len = chunk_lens.as_ref().map(|c| c[i]);
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
                        .into_primitive()
                        .vortex_unwrap()
                        .reinterpret_cast(PType::F16)
                        .into_array()),
                    PType::F32 => random_primitive::<f32>(u, *n, chunk_len),
                    PType::F64 => random_primitive::<f64>(u, *n, chunk_len),
                },
                DType::Utf8(n) => random_string(u, *n, chunk_len),
                DType::Binary(n) => random_bytes(u, *n, chunk_len),
                DType::Struct(sdt, n) => {
                    let first_array = sdt
                        .dtypes()
                        .first()
                        .map(|d| random_array(u, d, chunk_len))
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
                            sdt.dtypes()
                                .iter()
                                .skip(1)
                                .map(|d| random_array(u, d, Some(resolved_len))),
                        )
                        .collect::<Result<Vec<_>>>()?;
                    Ok(StructArray::try_new(
                        sdt.names().clone(),
                        children,
                        resolved_len,
                        random_validity(u, *n, resolved_len)?,
                    )
                    .vortex_unwrap()
                    .into_array())
                }
                DType::List(..) => {
                    todo!("List arrays are not implemented")
                }
                DType::Extension(..) => {
                    todo!("Extension arrays are not implemented")
                }
            }
        })
        .collect::<Result<Vec<_>>>()?;

    if chunks.len() == 1 {
        Ok(chunks.remove(0))
    } else {
        let dtype = chunks[0].dtype().clone();
        Ok(ChunkedArray::try_new(chunks, dtype)
            .vortex_unwrap()
            .into_array())
    }
}

fn split_number_into_parts(n: usize, parts: usize) -> Vec<usize> {
    let reminder = n % parts;
    let division = (n - reminder) / parts;
    iter::repeat(division)
        .take(parts - reminder)
        .chain(iter::repeat(division + 1).take(reminder))
        .collect()
}

fn random_string(
    u: &mut Unstructured,
    nullability: Nullability,
    len: Option<usize>,
) -> Result<ArrayData> {
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
) -> Result<ArrayData> {
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
) -> Result<ArrayData> {
    let v = arbitrary_vec_of_len::<T>(u, len)?;
    let validity = random_validity(u, nullability, v.len())?;
    Ok(PrimitiveArray::from_vec(v, validity).into_array())
}

fn random_bool(
    u: &mut Unstructured,
    nullability: Nullability,
    len: Option<usize>,
) -> Result<ArrayData> {
    let v = arbitrary_vec_of_len(u, len)?;
    let validity = random_validity(u, nullability, v.len())?;
    Ok(BoolArray::try_new(BooleanBuffer::from(v), validity)
        .vortex_expect("Validity length cannot mismatch")
        .into_array())
}

fn random_validity(u: &mut Unstructured, nullability: Nullability, len: usize) -> Result<Validity> {
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
