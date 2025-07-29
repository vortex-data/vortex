// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexResult};

use crate::arrays::{DecimalArray, DecimalVTable};
use crate::compute::{CastKernel, CastKernelAdapter};
use crate::validity::Validity;
use crate::{ArrayData, ArrayRef, IntoArray, register_kernel};

impl CastKernel for DecimalVTable {
    fn cast(&self, array: &DecimalArray, dtype: &DType) -> VortexResult<ArrayRef> {
        // We only support casting to the same decimal type with different nullability
        match (array.dtype(), dtype) {
            (DType::Decimal(from_precision_scale, from_nullability), DType::Decimal(to_precision_scale, to_nullability)) => {
                if from_precision_scale != to_precision_scale {
                    vortex_bail!(
                        "Cannot cast decimal({},{}) to decimal({},{})",
                        from_precision_scale.precision(),
                        from_precision_scale.scale(),
                        to_precision_scale.precision(),
                        to_precision_scale.scale()
                    );
                }

                // If nullability is the same, return self
                if from_nullability == to_nullability {
                    return Ok(array.to_array());
                }

                // Otherwise, create a new DecimalArray with the new nullability
                match (from_nullability, to_nullability) {
                    (Nullability::NonNullable, Nullability::Nullable) => {
                        // Cast from non-nullable to nullable - just change the validity
                        Ok(DecimalArray::new(
                            array.buffer().clone(),
                            *from_precision_scale,
                            array.validity_mask().to_nullable(),
                        ).to_array())
                    }
                    (Nullability::Nullable, Nullability::NonNullable) => {
                        // Cast from nullable to non-nullable - check if there are any nulls
                        if array.validity_mask().null_count() > 0 {
                            vortex_bail!("Cannot cast nullable decimal array with nulls to non-nullable");
                        }
                        Ok(DecimalArray::new(
                            array.buffer().clone(),
                            *from_precision_scale,
                            array.validity_mask().to_non_nullable(),
                        ).to_array())
                    }
                    _ => unreachable!("Nullability cases should be covered above"),
                }
            }
            _ => vortex_bail!(
                "Cannot cast {} to {}",
                array.dtype(),
                dtype
            ),
        }
    }
}

register_kernel!(CastKernelAdapter(DecimalVTable).lift());