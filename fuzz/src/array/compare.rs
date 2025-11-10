// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{BoolArray, NativeValue};
use vortex_array::compute::{Operator, scalar_cmp};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::BitBuffer;
use vortex_dtype::{DType, Nullability, match_each_decimal_value_type, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_scalar::Scalar;

pub fn compare_canonical_array(
    array: &dyn Array,
    value: &Scalar,
    operator: Operator,
) -> VortexResult<ArrayRef> {
    if value.is_null() {
        return Ok(BoolArray::from_bit_buffer(
            BitBuffer::new_unset(array.len()),
            Validity::AllInvalid,
        )
        .into_array());
    }

    let result_nullability = array.dtype().nullability() | value.dtype().nullability();

    match array.dtype() {
        DType::Bool(_) => {
            let bool = value
                .as_bool()
                .value()
                .vortex_expect("nulls handled before");
            Ok(compare_to(
                array
                    .to_bool()
                    .bit_buffer()
                    .iter()
                    .zip(array.validity_mask().to_bit_buffer().iter())
                    .map(|(b, v)| v.then_some(b)),
                bool,
                operator,
                result_nullability,
            ))
        }
        DType::Primitive(p, _) => {
            let primitive = value.as_primitive();
            let primitive_array = array.to_primitive();
            match_each_native_ptype!(p, |P| {
                let pval = primitive
                    .typed_value::<P>()
                    .vortex_expect("nulls handled before");
                Ok(compare_to(
                    primitive_array
                        .as_slice::<P>()
                        .iter()
                        .copied()
                        .zip(array.validity_mask().to_bit_buffer().iter())
                        .map(|(b, v)| v.then_some(NativeValue(b))),
                    NativeValue(pval),
                    operator,
                    result_nullability,
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
                Ok(compare_to(
                    buf.as_slice()
                        .iter()
                        .copied()
                        .zip(array.validity_mask().to_bit_buffer().iter())
                        .map(|(b, v)| v.then_some(b)),
                    dval,
                    operator,
                    result_nullability,
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
                &utf8_value,
                operator,
                result_nullability,
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
                &binary_value,
                operator,
                result_nullability,
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

fn compare_to<T: PartialOrd>(
    values: impl Iterator<Item = Option<T>>,
    cmp_value: T,
    operator: Operator,
    nullability: Nullability,
) -> ArrayRef {
    let eval_fn = |v| match operator {
        Operator::Eq => v == cmp_value,
        Operator::NotEq => v != cmp_value,
        Operator::Gt => v > cmp_value,
        Operator::Gte => v >= cmp_value,
        Operator::Lt => v < cmp_value,
        Operator::Lte => v <= cmp_value,
    };

    if !nullability.is_nullable() {
        BoolArray::from_iter(
            values
                .map(|val| val.vortex_expect("non nullable"))
                .map(eval_fn),
        )
        .into_array()
    } else {
        BoolArray::from_iter(values.map(|val| val.map(eval_fn))).into_array()
    }
}
