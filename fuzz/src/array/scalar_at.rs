// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::arrays::varbin_scalar;
use vortex_array::{Array, Canonical};
use vortex_dtype::{DType, match_each_native_ptype};
use vortex_error::{VortexResult, VortexUnwrap};
use vortex_scalar::{DecimalValue, Scalar, match_each_decimal_value_type};

/// Baseline implementation of scalar_at that works on canonical arrays.
/// This implementation manually extracts the scalar value from each canonical type
/// without using the scalar_at method, to serve as an independent baseline for testing.
pub fn scalar_at_canonical_array(canonical: Canonical, index: usize) -> VortexResult<Scalar> {
    Ok(match canonical {
        Canonical::Null(_array) => Scalar::null(DType::Null),
        Canonical::Bool(array) => {
            Scalar::bool(array.bit_buffer().value(index), array.dtype().nullability())
        }
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
            let list = array.list_elements_at(index);
            let children: Vec<Scalar> = (0..list.len())
                .map(|i| scalar_at_canonical_array(list.to_canonical(), i).vortex_unwrap())
                .collect();
            Scalar::list(
                Arc::new(list.dtype().clone()),
                children,
                array.dtype().nullability(),
            )
        }
        Canonical::FixedSizeList(array) => {
            let list = array.fixed_size_list_elements_at(index);
            let children: Vec<Scalar> = (0..list.len())
                .map(|i| scalar_at_canonical_array(list.to_canonical(), i).vortex_unwrap())
                .collect();
            Scalar::fixed_size_list(list.dtype().clone(), children, array.dtype().nullability())
        }
        Canonical::Struct(array) => {
            let field_scalars: Vec<Scalar> = array
                .fields()
                .iter()
                .map(|field| scalar_at_canonical_array(field.to_canonical(), index).vortex_unwrap())
                .collect();
            Scalar::struct_(array.dtype().clone(), field_scalars)
        }
        Canonical::Extension(array) => {
            let storage_scalar = scalar_at_canonical_array(array.storage().to_canonical(), index)?;
            Scalar::extension(array.ext_dtype().clone(), storage_scalar)
        }
    })
}
