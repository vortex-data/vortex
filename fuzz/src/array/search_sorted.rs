// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

// Count the insertion boundary directly instead of reusing production binary search.
fn linear_search(
    comparisons: impl Iterator<Item = Ordering>,
    side: SearchSortedSide,
) -> SearchResult {
    let mut less = 0;
    let mut equal = 0;
    for comparison in comparisons {
        match comparison {
            Ordering::Less => less += 1,
            Ordering::Equal => equal += 1,
            Ordering::Greater => {}
        }
    }
    let index = less
        + if side == SearchSortedSide::Right {
            equal
        } else {
            0
        };
    if equal == 0 {
        SearchResult::NotFound(index)
    } else {
        SearchResult::Found(index)
    }
}

pub fn search_sorted_canonical_array(
    array: &ArrayRef,
    scalar: &Scalar,
    side: SearchSortedSide,
    ctx: &mut ExecutionCtx,
) -> VortexResult<SearchResult> {
    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.clone().execute::<BoolArray>(ctx)?;
            let validity = bool_array
                .as_ref()
                .validity()?
                .execute_mask(bool_array.as_ref().len(), ctx)?
                .to_bit_buffer();
            let opt_values = bool_array
                .to_bit_buffer()
                .iter()
                .zip(validity.iter())
                .map(|(b, v)| v.then_some(b))
                .collect::<Vec<_>>();
            let to_find = scalar.try_into()?;
            Ok(linear_search(
                opt_values.iter().map(|value| value.cmp(&Some(to_find))),
                side,
            ))
        }
        DType::Primitive(p, _) => {
            let primitive_array = array.clone().execute::<PrimitiveArray>(ctx)?;
            let validity = primitive_array
                .as_ref()
                .validity()?
                .execute_mask(primitive_array.as_ref().len(), ctx)?
                .to_bit_buffer();
            match_each_native_ptype!(p, |P| {
                let opt_values = primitive_array
                    .as_slice::<P>()
                    .iter()
                    .copied()
                    .zip(validity.iter())
                    .map(|(b, v)| v.then_some(b))
                    .collect::<Vec<_>>();
                let to_find: P = scalar.try_into()?;
                Ok(linear_search(
                    opt_values.iter().map(|value| match value {
                        None => Ordering::Less,
                        Some(value) => value.total_compare(to_find),
                    }),
                    side,
                ))
            })
        }
        DType::Decimal(d, _) => {
            let decimal_array = array.clone().execute::<DecimalArray>(ctx)?;
            let validity = decimal_array
                .as_ref()
                .validity()?
                .execute_mask(decimal_array.as_ref().len(), ctx)?
                .to_bit_buffer();
            match_each_decimal_value_type!(decimal_array.values_type(), |D| {
                let buf = decimal_array.buffer::<D>();
                let opt_values = buf
                    .as_slice()
                    .iter()
                    .copied()
                    .zip(validity.iter())
                    .map(|(b, v)| v.then_some(b))
                    .collect::<Vec<_>>();
                let to_find: D = scalar
                    .as_decimal()
                    .decimal_value()
                    .map(|v| {
                        v.cast::<D>().ok_or_else(|| {
                            vortex_err!("cannot cast value {v} to decimal value type {d}")
                        })
                    })
                    .transpose()?
                    .ok_or_else(|| vortex_err!("unexpected null scalar"))?;
                Ok(linear_search(
                    opt_values.iter().map(|value| value.cmp(&Some(to_find))),
                    side,
                ))
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.clone().execute::<VarBinViewArray>(ctx)?;
            let mask = utf8.validity()?.execute_mask(utf8.len(), ctx)?;
            let opt_values = (0..utf8.len())
                .map(|i| mask.value(i).then(|| utf8.bytes_at(i).to_vec()))
                .collect::<Vec<_>>();
            let to_find = if matches!(array.dtype(), DType::Utf8(_)) {
                BufferString::try_from(scalar)?.as_str().as_bytes().to_vec()
            } else {
                ByteBuffer::try_from(scalar)?.to_vec()
            };
            let to_find = Some(to_find);
            Ok(linear_search(
                opt_values.iter().map(|value| value.cmp(&to_find)),
                side,
            ))
        }
        DType::List(..) | DType::FixedSizeList(..) | DType::Struct(..) => {
            let scalar_vals = (0..array.len())
                .map(|i| array.execute_scalar(i, ctx))
                .collect::<VortexResult<Vec<_>>>()?;
            scalar_vals.search_sorted(&scalar.cast(array.dtype())?, side)
        }
        d @ (DType::Null
        | DType::Map(..)
        | DType::Union(..)
        | DType::Variant(_)
        | DType::Extension(_)) => {
            unreachable!("DType {d} not supported for fuzzing")
        }
    }
}
