// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt::Debug;

use vortex_array::ArrayRef;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::search_sorted::IndexOrd;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

struct SearchNullableSlice<T>(Vec<Option<T>>);

impl<T: PartialOrd + Debug> IndexOrd<Option<T>> for SearchNullableSlice<T> {
    fn index_cmp(&self, idx: usize, elem: &Option<T>) -> VortexResult<Option<Ordering>> {
        // SAFETY: Used in search_sorted_by same as the standard library. The search_sorted ensures idx is in bounds
        Ok(unsafe { self.0.get_unchecked(idx) }.partial_cmp(elem))
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

struct SearchPrimitiveSlice<T>(Vec<Option<T>>);

impl<T: NativePType> IndexOrd<Option<T>> for SearchPrimitiveSlice<T> {
    fn index_cmp(&self, idx: usize, elem: &Option<T>) -> VortexResult<Option<Ordering>> {
        match elem {
            None => unreachable!("Can't search for None"),
            Some(v) => {
                // SAFETY: Used in search_sorted_by same as the standard library. The search_sorted ensures idx is in bounds
                Ok(match unsafe { self.0.get_unchecked(idx) } {
                    None => Some(Ordering::Less),
                    Some(i) => Some(i.total_compare(*v)),
                })
            }
        }
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

pub fn search_sorted_canonical_array(
    array: &ArrayRef,
    scalar: &Scalar,
    side: SearchSortedSide,
) -> VortexResult<SearchResult> {
    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.to_bool();
            let validity = bool_array
                .as_ref()
                .validity()?
                .to_mask(
                    bool_array.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .to_bit_buffer();
            let opt_values = bool_array
                .to_bit_buffer()
                .iter()
                .zip(validity.iter())
                .map(|(b, v)| v.then_some(b))
                .collect::<Vec<_>>();
            let to_find = scalar.try_into()?;
            SearchNullableSlice(opt_values).search_sorted(&Some(to_find), side)
        }
        DType::Primitive(p, _) => {
            let primitive_array = array.to_primitive();
            let validity = primitive_array
                .as_ref()
                .validity()?
                .to_mask(
                    primitive_array.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
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
                SearchPrimitiveSlice(opt_values).search_sorted(&Some(to_find), side)
            })
        }
        DType::Decimal(d, _) => {
            let decimal_array = array.to_decimal();
            let validity = decimal_array
                .as_ref()
                .validity()?
                .to_mask(
                    decimal_array.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
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
                SearchNullableSlice(opt_values).search_sorted(&Some(to_find), side)
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.to_varbinview();
            let opt_values =
                utf8.with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>());
            let to_find = if matches!(array.dtype(), DType::Utf8(_)) {
                BufferString::try_from(scalar)?.as_str().as_bytes().to_vec()
            } else {
                ByteBuffer::try_from(scalar)?.to_vec()
            };
            SearchNullableSlice(opt_values).search_sorted(&Some(to_find), side)
        }
        DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..) => {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let scalar_vals = (0..array.len())
                .map(|i| array.execute_scalar(i, &mut ctx))
                .collect::<VortexResult<Vec<_>>>()?;
            scalar_vals.search_sorted(&scalar.cast(array.dtype())?, side)
        }
        d @ (DType::Null | DType::Extension(_) | DType::Variant(_)) => {
            unreachable!("DType {d} not supported for fuzzing")
        }
    }
}
