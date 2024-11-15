use std::cmp::Ordering;

use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{BoolArray, PrimitiveArray, VarBinViewArray};
use vortex_array::compute::unary::scalar_at;
use vortex_array::validity::ArrayValidity;
use vortex_array::{Array, ArrayDType, IntoArray, IntoArrayVariant};
use vortex_dtype::{match_each_native_ptype, DType, NativePType};

use crate::take::take_canonical_array;

pub fn sort_canonical_array(array: &Array) -> Array {
    match array.dtype() {
        DType::Bool(_) => {
            let bool_array = array.clone().into_bool().unwrap();
            let mut opt_values = bool_array
                .boolean_buffer()
                .iter()
                .zip(
                    bool_array
                        .logical_validity()
                        .into_array()
                        .into_bool()
                        .unwrap()
                        .boolean_buffer()
                        .iter(),
                )
                .map(|(b, v)| v.then_some(b))
                .collect::<Vec<_>>();
            sort_opt_slice(&mut opt_values);
            BoolArray::from_iter(opt_values).into_array()
        }
        DType::Primitive(p, _) => {
            let primitive_array = array.clone().into_primitive().unwrap();
            match_each_native_ptype!(p, |$P| {
                let mut opt_values = primitive_array
                    .maybe_null_slice::<$P>()
                    .iter()
                    .copied()
                    .zip(
                        primitive_array
                            .logical_validity()
                            .into_array()
                            .into_bool()
                            .unwrap()
                            .boolean_buffer()
                            .iter(),
                    )
                    .map(|(p, v)| v.then_some(p))
                    .collect::<Vec<_>>();
                sort_primitive_slice(&mut opt_values);
                PrimitiveArray::from_nullable_vec(opt_values).into_array()
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let utf8 = array.clone().into_varbinview().unwrap();
            let mut opt_values = utf8
                .with_iterator(|iter| iter.map(|v| v.map(|u| u.to_vec())).collect::<Vec<_>>())
                .unwrap();
            sort_opt_slice(&mut opt_values);
            VarBinViewArray::from_iter(opt_values, array.dtype().clone()).into_array()
        }
        DType::Struct(..) => {
            let mut sort_indices = (0..array.len()).collect::<Vec<_>>();
            sort_indices.sort_by(|a, b| {
                scalar_at(array, *a)
                    .unwrap()
                    .partial_cmp(&scalar_at(array, *b).unwrap())
                    .unwrap()
            });
            take_canonical_array(array, &sort_indices)
        }
        _ => unreachable!("Not a canonical array"),
    }
}

fn sort_primitive_slice<T: NativePType>(s: &mut [Option<T>]) {
    s.sort_by(|a, b| match (a, b) {
        (Some(v), Some(w)) => v.total_compare(*w),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
    });
}

/// Reverse sorting of Option<T> such that None is last (Greatest)
fn sort_opt_slice<T: Ord>(s: &mut [Option<T>]) {
    s.sort_by(|a, b| match (a, b) {
        (Some(v), Some(w)) => v.cmp(w),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
    });
}
