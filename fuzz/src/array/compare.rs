// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::scalar_cmp;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

pub fn compare_canonical_array(
    array: &ArrayRef,
    value: &Scalar,
    operator: CompareOperator,
) -> ArrayRef {
    if value.is_null() {
        return BoolArray::new(BitBuffer::new_unset(array.len()), Validity::AllInvalid)
            .into_array();
    }

    let result_nullability = array.dtype().nullability() | value.dtype().nullability();

    match array.dtype() {
        DType::Bool(_) => {
            let bool = value
                .as_bool()
                .value()
                .vortex_expect("nulls handled before");
            compare_to(
                array
                    .to_bool()
                    .to_bit_buffer()
                    .iter()
                    .zip(
                        array
                            .validity()
                            .vortex_expect("validity_mask")
                            .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())
                            .vortex_expect("Failed to compute validity mask")
                            .to_bit_buffer()
                            .iter(),
                    )
                    .map(|(b, v)| v.then_some(b)),
                bool,
                operator,
                result_nullability,
            )
        }
        DType::Primitive(p, _) => {
            let primitive = value.as_primitive();
            let primitive_array = array.to_primitive();
            match_each_native_ptype!(p, |P| {
                let pval = primitive
                    .typed_value::<P>()
                    .vortex_expect("nulls handled before");
                compare_to(
                    primitive_array
                        .as_slice::<P>()
                        .iter()
                        .copied()
                        .zip(
                            array
                                .validity()
                                .vortex_expect("validity_mask")
                                .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())
                                .vortex_expect("Failed to compute validity mask")
                                .to_bit_buffer()
                                .iter(),
                        )
                        .map(|(b, v)| v.then_some(NativeValue(b))),
                    NativeValue(pval),
                    operator,
                    result_nullability,
                )
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
                    .unwrap_or_else(|| vortex_panic!("todo: handle upcast of decimal array"));
                let buf = decimal_array.buffer::<D>();
                compare_to(
                    buf.as_slice()
                        .iter()
                        .copied()
                        .zip(
                            array
                                .validity()
                                .vortex_expect("validity_mask")
                                .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())
                                .vortex_expect("Failed to compute validity mask")
                                .to_bit_buffer()
                                .iter(),
                        )
                        .map(|(b, v)| v.then_some(b)),
                    dval,
                    operator,
                    result_nullability,
                )
            })
        }
        DType::Utf8(_) => array.to_varbinview().with_iterator(|iter| {
            let utf8_value = value.as_utf8();
            compare_to(
                iter.map(|v| v.map(|b| unsafe { str::from_utf8_unchecked(b) })),
                utf8_value.value().vortex_expect("nulls handled before"),
                operator,
                result_nullability,
            )
        }),
        DType::Binary(_) => array.to_varbinview().with_iterator(|iter| {
            let binary_value = value.as_binary();
            compare_to(
                // Don't understand the lifetime problem here but identity map makes it go away
                #[expect(clippy::map_identity)]
                iter.map(|v| v),
                binary_value.value().vortex_expect("nulls handled before"),
                operator,
                result_nullability,
            )
        }),
        DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..) => {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let scalar_vals: Vec<Scalar> = (0..array.len())
                .map(|i| array.execute_scalar(i, &mut ctx).vortex_expect("scalar_at"))
                .collect();
            BoolArray::from_iter(scalar_vals.iter().map(|v| {
                scalar_cmp(v, value, operator)
                    .vortex_expect("tried to compare different typed scalars")
                    .as_bool()
                    .value()
            }))
            .into_array()
        }
        d @ (DType::Null | DType::Extension(_) | DType::Variant(_)) => {
            unreachable!("DType {d} not supported for fuzzing")
        }
    }
}

fn compare_to<T: PartialOrd>(
    values: impl Iterator<Item = Option<T>>,
    cmp_value: T,
    operator: CompareOperator,
    nullability: Nullability,
) -> ArrayRef {
    let eval_fn = |v| match operator {
        CompareOperator::Eq => v == cmp_value,
        CompareOperator::NotEq => v != cmp_value,
        CompareOperator::Gt => v > cmp_value,
        CompareOperator::Gte => v >= cmp_value,
        CompareOperator::Lt => v < cmp_value,
        CompareOperator::Lte => v <= cmp_value,
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
