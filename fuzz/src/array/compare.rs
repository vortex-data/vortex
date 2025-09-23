// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::ops::Deref;

use arrow_buffer::BooleanBuffer;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::BoolArray;
use vortex_array::compute::{Operator, scalar_cmp};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_scalar::{NativeDecimalType, Scalar, match_each_decimal_value_type};

pub fn compare_canonical_array(
    array: &dyn Array,
    value: &Scalar,
    operator: Operator,
) -> VortexResult<ArrayRef> {
    if value.is_null() {
        return Ok(BoolArray::from_bool_buffer(
            BooleanBuffer::new_unset(array.len()),
            Validity::AllInvalid,
        )
        .into_array());
    }

    match array.dtype() {
        DType::Bool(_) => {
            let bool = value
                .as_bool()
                .value()
                .vortex_expect("nulls handled before");
            Ok(compare_to(
                array
                    .to_bool()
                    .boolean_buffer()
                    .iter()
                    .zip(array.validity_mask().to_boolean_buffer().iter())
                    .map(|(b, v)| v.then_some(b)),
                bool,
                operator,
            ))
        }
        DType::Primitive(p, _) => {
            let primitive = value.as_primitive();
            let primitive_array = array.to_primitive();
            match_each_native_ptype!(p, |P| {
                let pval = primitive
                    .typed_value::<P>()
                    .vortex_expect("nulls handled before");
                Ok(compare_native_ptype(
                    primitive_array
                        .as_slice::<P>()
                        .iter()
                        .copied()
                        .zip(array.validity_mask().to_boolean_buffer().iter())
                        .map(|(b, v)| v.then_some(b)),
                    pval,
                    operator,
                ))
            })
        }
        DType::Decimal(..) => {
            let decimal = value.as_decimal();
            let decimal_array = array.to_decimal();
            match_each_decimal_value_type!(decimal_array.values_type(), |D| {
                let dval = decimal
                    .decimal_value()
                    .vortex_expect("nulls handled before")
                    .cast::<D>()
                    .ok_or_else(|| vortex_err!("todo: handle upcast of decimal array"))?;
                let buf = decimal_array.buffer::<D>();
                Ok(compare_native_decimal_type(
                    buf.as_slice()
                        .iter()
                        .copied()
                        .zip(array.validity_mask().to_boolean_buffer().iter())
                        .map(|(b, v)| v.then_some(b)),
                    dval,
                    operator,
                ))
            })
        }
        DType::Utf8(_) => array.to_varbinview().with_iterator(|iter| {
            let utf8_value = value
                .as_utf8()
                .value()
                .vortex_expect("nulls handled before");
            compare_to(
                iter.map(|v| v.map(|b| unsafe { str::from_utf8_unchecked(b) })),
                utf8_value.deref(),
                operator,
            )
        }),
        DType::Binary(_) => array.to_varbinview().with_iterator(|iter| {
            let binary_value = value
                .as_binary()
                .value()
                .vortex_expect("nulls handled before");
            compare_to(
                // Don't understand the lifetime problem here but identity map makes it go away
                #[allow(clippy::map_identity)]
                iter.map(|v| v),
                binary_value.deref(),
                operator,
            )
        }),
        DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..) => {
            let scalar_vals: Vec<Scalar> = (0..array.len()).map(|i| array.scalar_at(i)).collect();
            Ok(BoolArray::from_iter(
                scalar_vals
                    .iter()
                    .map(|v| scalar_cmp(v, value, operator).as_bool().value()),
            )
            .into_array())
        }
        d @ (DType::Null | DType::Extension(_)) => {
            unreachable!("DType {d} not supported for fuzzing")
        }
    }
}

fn compare_to<T: PartialOrd + PartialEq + Debug>(
    values: impl Iterator<Item = Option<T>>,
    cmp_value: T,
    operator: Operator,
) -> ArrayRef {
    BoolArray::from_iter(values.map(|val| {
        val.map(|v| match operator {
            Operator::Eq => v == cmp_value,
            Operator::NotEq => v != cmp_value,
            Operator::Gt => v > cmp_value,
            Operator::Gte => v >= cmp_value,
            Operator::Lt => v < cmp_value,
            Operator::Lte => v <= cmp_value,
        })
    }))
    .into_array()
}

fn compare_native_ptype<T: NativePType>(
    values: impl Iterator<Item = Option<T>>,
    cmp_value: T,
    operator: Operator,
) -> ArrayRef {
    BoolArray::from_iter(values.map(|val| {
        val.map(|v| match operator {
            Operator::Eq => v.is_eq(cmp_value),
            Operator::NotEq => !v.is_eq(cmp_value),
            Operator::Gt => v.is_gt(cmp_value),
            Operator::Gte => v.is_ge(cmp_value),
            Operator::Lt => v.is_lt(cmp_value),
            Operator::Lte => v.is_le(cmp_value),
        })
    }))
    .into_array()
}

fn compare_native_decimal_type<D: NativeDecimalType>(
    values: impl Iterator<Item = Option<D>>,
    cmp_value: D,
    operator: Operator,
) -> ArrayRef {
    BoolArray::from_iter(values.map(|val| {
        val.map(|v| match operator {
            Operator::Eq => v == cmp_value,
            Operator::NotEq => v != cmp_value,
            Operator::Gt => v > cmp_value,
            Operator::Gte => v >= cmp_value,
            Operator::Lt => v < cmp_value,
            Operator::Lte => v <= cmp_value,
        })
    }))
    .into_array()
}
