// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter};
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail};

use crate::{FSSTViewArray, FSSTViewVTable};

impl CastKernel for FSSTViewVTable {
    fn cast(&self, array: &FSSTViewArray, target: &DType) -> VortexResult<Option<ArrayRef>> {
        // Types of casting supported:
        // NonNull -> Nullable
        // AllValid -> NonNull
        // Other casts delegate to the canonical implementation

        // TODO(aduffy): support casting from Binary -> Utf8 through validation
        if array.dtype().is_binary() && target.is_utf8() {
            return Ok(None);
        }

        let old_nullability = array.dtype.nullability();
        let new_nullability = target.nullability();

        match (old_nullability, new_nullability) {
            (Nullability::Nullable, Nullability::Nullable)
            | (Nullability::NonNullable, Nullability::NonNullable) => {
                // SAFETY: Trivially satisfied. Validity is unmodified b/c nullability is unmodified
                unsafe {
                    Ok(Some(
                        FSSTViewArray::new_unchecked(
                            array.views.clone(),
                            array.fsst_buffer.clone(),
                            array.symbols.clone(),
                            array.symbol_lengths.clone(),
                            array.compressed_offsets.clone(),
                            array.uncompressed_offsets.clone(),
                            target.clone(),
                            array.validity.clone(),
                        )
                        .into_array(),
                    ))
                }
            }
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
                            target.clone(),
                            Validity::AllValid,
                        )
                        .into_array(),
                    ))
                }
            }
            (Nullability::Nullable, Nullability::NonNullable) => {
                // Casting nullable -> non-nullable requires that there are no nulls in the data
                if array.invalid_count() > 0 {
                    vortex_bail!(
                        "Failed to cast {} to {target}: array contains nulls",
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
                            target.clone(),
                            Validity::NonNullable,
                        )
                        .into_array(),
                    ))
                }
            }
        }
    }
}

register_kernel!(CastKernelAdapter(FSSTViewVTable).lift());
