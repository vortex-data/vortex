// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
#[expect(deprecated)]
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::array::take_canonical_array_non_nullable_indices;

pub fn sort_canonical_array(array: &ArrayRef) -> VortexResult<ArrayRef> {
    match array.dtype() {
        DType::Bool(_) => {
            #[expect(deprecated)]
            let bool_array = array.to_bool();
            let mut opt_values = bool_array
                .to_bit_buffer()
                .iter()
                .zip(
                    bool_array
                        .as_ref()
                        .validity()?
                        .to_mask(
                            bool_array.as_ref().len(),
                            &mut LEGACY_SESSION.create_execution_ctx(),
                        )?
                        .to_bit_buffer()
                        .iter(),
                )
                .map(|(b, v)| v.then_some(b))
                .collect::<Vec<_>>();
            opt_values.sort();
            Ok(BoolArray::from_iter(opt_values).into_array())
        }
        DType::Primitive(p, _) => {
            #[expect(deprecated)]
            let primitive_array = array.to_primitive();
            match_each_native_ptype!(p, |P| {
                let mut opt_values = primitive_array
                    .as_slice::<P>()
                    .iter()
                    .copied()
                    .zip(
                        primitive_array
                            .as_ref()
                            .validity()?
                            .to_mask(
                                primitive_array.as_ref().len(),
                                &mut LEGACY_SESSION.create_execution_ctx(),
                            )?
                            .to_bit_buffer()
                            .iter(),
                    )
                    .map(|(p, v)| v.then_some(p))
                    .collect::<Vec<_>>();
                sort_primitive_slice(&mut opt_values);
                Ok(PrimitiveArray::from_option_iter(opt_values).into_array())
            })
        }
        DType::Decimal(d, _) => {
            #[expect(deprecated)]
            let decimal_array = array.to_decimal();
            match_each_decimal_value_type!(decimal_array.values_type(), |D| {
                let buf = decimal_array.buffer::<D>();
                let mut opt_values = buf
                    .as_slice()
                    .iter()
                    .copied()
                    .zip(
                        decimal_array
                            .as_ref()
                            .validity()?
                            .to_mask(
                                decimal_array.as_ref().len(),
                                &mut LEGACY_SESSION.create_execution_ctx(),
                            )?
                            .to_bit_buffer()
                            .iter(),
                    )
                    .map(|(p, v)| v.then_some(p))
                    .collect::<Vec<_>>();
                opt_values.sort();
                Ok(DecimalArray::from_option_iter(opt_values, *d).into_array())
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            #[expect(deprecated)]
            let utf8 = array.to_varbinview();
            let mut opt_values =
                utf8.with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>());
            opt_values.sort();
            Ok(VarBinViewArray::from_iter(opt_values, array.dtype().clone()).into_array())
        }
        DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..) => {
            let mut sort_indices = (0..array.len()).collect::<Vec<_>>();
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            sort_indices.sort_by(|a, b| {
                let lhs = array
                    .execute_scalar(*a, &mut ctx)
                    .vortex_expect("scalar_at");
                let rhs = array
                    .execute_scalar(*b, &mut ctx)
                    .vortex_expect("scalar_at");
                lhs.partial_cmp(&rhs)
                    .vortex_expect("must be a valid comparison")
            });
            take_canonical_array_non_nullable_indices(array, &sort_indices)
        }
        d @ (DType::Null | DType::Extension(_) | DType::Variant(_)) => {
            unreachable!("DType {d} not supported for fuzzing")
        }
    }
}

fn sort_primitive_slice<T: NativePType>(values: &mut [Option<T>]) {
    values.sort_by(|a, b| match (a, b) {
        (Some(sa), Some(sb)) => sa.total_compare(*sb),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
    });
}
