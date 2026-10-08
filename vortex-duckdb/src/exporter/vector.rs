// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::mask::Mask;

use crate::cpp::duckdb_vx_vector_set_all_valid;
use crate::duckdb::ValidityData;
use crate::duckdb::Value;
use crate::duckdb::VectorRef;
use crate::exporter::copy_from_slice;

impl VectorRef {
    /// Sets validity from the selected range of `mask`.
    ///
    /// Returns whether all selected values are null.
    ///
    /// # Safety
    ///
    /// - `offset + len` must not exceed `mask.len()`.
    /// - `len` must not exceed the vector capacity.
    pub unsafe fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool {
        unsafe { self.set_validity_zero_copy(mask, offset, len, None) }
    }

    /// Sets validity from `mask`, using `zero_copy` when its bitmap and `offset` are `u64`-aligned.
    ///
    /// Returns whether all selected values are null.
    ///
    /// # Safety
    ///
    /// - `offset + len` must not exceed `mask.len()`.
    /// - `len` must not exceed the vector capacity.
    /// - A supplied `zero_copy` value must point to the start of the `Mask::Values` bitmap in
    ///   `mask`. The pointer must be aligned for `u64`, and the buffer must contain only complete
    ///   `u64` words.
    pub(super) unsafe fn set_validity_zero_copy(
        &mut self,
        mask: &Mask,
        offset: usize,
        len: usize,
        zero_copy: Option<&ValidityData>,
    ) -> bool {
        let values = match mask {
            Mask::AllTrue(_) => {
                self.set_all_true_validity();
                return false;
            }
            Mask::AllFalse(_) => {
                self.set_all_false_validity();
                return true;
            }
            Mask::Values(values) => values,
        };

        let true_count = values.bit_buffer().count_range(offset, offset + len);
        if true_count == 0 {
            self.set_all_false_validity();
            return true;
        }

        if true_count == len {
            self.set_all_true_validity();
        } else if let Some(validity_data) = zero_copy.filter(|_| offset.is_multiple_of(64)) {
            let u64_offset = offset / 64;

            // SAFETY:
            // - `zero_copy_validity` points `data_ptr` to an aligned buffer of complete
            //   `u64` words.
            // - `ValidityExporter::export` bounds the selected range to the mask, and this
            //   branch requires a `u64`-aligned offset, so the buffer contains every word.
            // - `shared_buffer` keeps the bitmap alive while DuckDB reads it.
            unsafe { self.set_validity_data(u64_offset, len, validity_data) };
        } else {
            debug_assert!(zero_copy.is_none() || !offset.is_multiple_of(64));
            let source = values.bit_buffer().inner().as_slice();
            copy_from_slice(
                unsafe { self.ensure_validity_slice(len) },
                source,
                offset,
                len,
            );
        }
        false
    }

    pub fn set_all_true_validity(&mut self) {
        unsafe { duckdb_vx_vector_set_all_valid(self.as_ptr()) };
    }

    pub fn set_all_false_validity(&mut self) {
        self.reference_value(&Value::null(&self.logical_type()));
    }
}
