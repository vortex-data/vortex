// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter};
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail};

use crate::{FSSTViewArray, FSSTViewVTable};

impl CastKernel for FSSTViewVTable {
    fn cast(&self, array: &FSSTViewArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Types of casting supported:
        // NonNull -> Nullable
        // AllValid -> NonNull
        // Other casts delegate to the canonical implementation

        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }

        let old_nullability = array.dtype.nullability();
        let new_nullability = dtype.nullability();

        match (old_nullability, new_nullability) {
            // Converting from non-null -> nullable means Validity becomes AllValid
            (Nullability::NonNullable, Nullability::Nullable) => {
                // SAFETY: changing validity to Nullable is trivial
                unsafe {
                    Ok(Some(
                        FSSTViewArray::new_unchecked(
                            array.views.clone(),
                            array.fsst_buffer.clone(),
                            array.symbols.clone(),
                            array.symbol_lengths.clone(),
                            array.compressed_offsets.clone(),
                            array.uncompressed_offsets.clone(),
                            dtype.clone(),
                            Validity::AllValid,
                        )
                        .into_array(),
                    ))
                }
            }
            (Nullability::Nullable, Nullability::NonNullable) => {
                if array.validity.null_count(array.len())? > 0 {
                    vortex_bail!(
                        "Failed to cast {} to {dtype}: array contains nulls",
                        array.dtype()
                    );
                }

                // SAFETY: changing validity to NonNullable when there are no nulls is trivial
                unsafe {
                    Ok(Some(
                        FSSTViewArray::new_unchecked(
                            array.views.clone(),
                            array.fsst_buffer.clone(),
                            array.symbols.clone(),
                            array.symbol_lengths.clone(),
                            array.compressed_offsets.clone(),
                            array.uncompressed_offsets.clone(),
                            dtype.clone(),
                            Validity::NonNullable,
                        )
                        .into_array(),
                    ))
                }
            }
            _ => Ok(None),
        }
    }
}

register_kernel!(CastKernelAdapter(FSSTViewVTable).lift());
