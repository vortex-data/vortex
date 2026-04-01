// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::dtype::DType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

/// Baseline implementation of scalar_at that works on canonical arrays.
/// This implementation manually extracts the scalar value from each canonical type
/// without using the scalar_at method, to serve as an independent baseline for testing.
pub fn scalar_at_canonical_array(canonical: Canonical, index: usize) -> VortexResult<Scalar> {
    let canonical_ref = canonical.clone().into_array();
    if canonical_ref.is_invalid(index)? {
        return Ok(Scalar::null(canonical_ref.dtype().clone()));
    }
    Ok(match canonical {
        Canonical::Null(_array) => Scalar::null(DType::Null),
        Canonical::Bool(array) => Scalar::bool(
            array.to_bit_buffer().value(index),
            array.dtype().nullability(),
        ),
        Canonical::Primitive(array) => {
            match_each_native_ptype!(array.ptype(), |T| {
                Scalar::primitive(array.as_slice::<T>()[index], array.dtype().nullability())
            })
        }
        Canonical::Decimal(array) => {
            match_each_decimal_value_type!(array.values_type(), |D| {
                Scalar::decimal(
                    DecimalValue::from(array.buffer::<D>()[index]),
                    array.decimal_dtype(),
                    array.dtype().nullability(),
                )
            })
        }
        Canonical::VarBinView(array) => varbin_scalar(array.bytes_at(index), array.dtype()),
        Canonical::List(array) => {
            let list = array.list_elements_at(index)?;
            let children: Vec<Scalar> = (0..list.len())
                .map(|i| {
                    scalar_at_canonical_array(
                        list.to_canonical()
                            .vortex_expect("to_canonical should succeed in fuzz test"),
                        i,
                    )
                    .vortex_expect("scalar_at_canonical_array should succeed in fuzz test")
                })
                .collect();
            Scalar::list(
                Arc::new(list.dtype().clone()),
                children,
                array.dtype().nullability(),
            )
        }
        Canonical::FixedSizeList(array) => {
            let list = array.fixed_size_list_elements_at(index)?;
            let children: Vec<Scalar> = (0..list.len())
                .map(|i| {
                    scalar_at_canonical_array(
                        list.to_canonical()
                            .vortex_expect("to_canonical should succeed in fuzz test"),
                        i,
                    )
                    .vortex_expect("scalar_at_canonical_array should succeed in fuzz test")
                })
                .collect();
            Scalar::fixed_size_list(list.dtype().clone(), children, array.dtype().nullability())
        }
        Canonical::Struct(array) => {
            let field_scalars: Vec<Scalar> = array
                .iter_unmasked_fields()
                .map(|field| {
                    scalar_at_canonical_array(
                        field
                            .to_canonical()
                            .vortex_expect("to_canonical should succeed in fuzz test"),
                        index,
                    )
                    .vortex_expect("scalar_at_canonical_array should succeed in fuzz test")
                })
                .collect();
            Scalar::struct_(array.dtype().clone(), field_scalars)
        }
        Canonical::Extension(array) => {
            let storage_scalar =
                scalar_at_canonical_array(array.storage_array().to_canonical()?, index)?;
            Scalar::extension_ref(array.ext_dtype().clone(), storage_scalar)
        }
        Canonical::Variant(_) => unreachable!("Variant arrays are not fuzzed"),
    })
}
