use std::cmp::Ordering;

use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{BoolArray, PrimitiveArray, VarBinViewArray};
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap};

use crate::take::take_canonical_array;

pub fn sort_canonical_array(array: &dyn Array) -> VortexResult<ArrayRef> {
    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.to_bool()?;
            let mut opt_values = bool_array
                .boolean_buffer()
                .iter()
                .zip(bool_array.validity_mask()?.to_boolean_buffer().iter())
                .map(|(b, v)| v.then_some(b))
                .collect::<Vec<_>>();
            opt_values.sort();
            Ok(BoolArray::from_iter(opt_values).into_array())
        }
        DType::Primitive(p, _) => {
            let primitive_array = array.to_primitive()?;
            match_each_native_ptype!(p, |P| {
                let mut opt_values = primitive_array
                    .as_slice::<P>()
                    .iter()
                    .copied()
                    .zip(primitive_array.validity_mask()?.to_boolean_buffer().iter())
                    .map(|(p, v)| v.then_some(p))
                    .collect::<Vec<_>>();
                sort_primitive_slice(&mut opt_values);
                Ok(PrimitiveArray::from_option_iter(opt_values).into_array())
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.to_varbinview()?;
            let mut opt_values =
                utf8.with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>())?;
            opt_values.sort();
            Ok(VarBinViewArray::from_iter(opt_values, array.dtype().clone()).into_array())
        }
        DType::Struct(..) => {
            let mut sort_indices = (0..array.len()).collect::<Vec<_>>();
            sort_indices.sort_by(|a, b| {
                array
                    .scalar_at(*a)
                    .vortex_unwrap()
                    .partial_cmp(&array.scalar_at(*b).vortex_unwrap())
                    .vortex_expect("must be a valid comparison")
            });
            take_canonical_array(array, &sort_indices)
        }
        DType::List(..) => {
            let mut sort_indices = (0..array.len()).collect::<Vec<_>>();
            sort_indices.sort_by(|a, b| {
                array
                    .scalar_at(*a)
                    .vortex_unwrap()
                    .partial_cmp(&array.scalar_at(*b).vortex_unwrap())
                    .vortex_expect("must be a valid comparison")
            });
            take_canonical_array(array, &sort_indices)
        }
        d => unreachable!("DType {d} not supported for fuzzing"),
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
