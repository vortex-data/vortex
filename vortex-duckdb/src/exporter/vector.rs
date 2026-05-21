// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::mask::Mask;

use crate::cpp::duckdb_vx_vector_set_all_valid;
use crate::duckdb::ValidityData;
use crate::duckdb::Value;
use crate::duckdb::VectorRef;
use crate::exporter::copy_from_slice;

impl VectorRef {
    /// Returns true if all values are null (caller can skip data export).
    pub unsafe fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool {
        unsafe { self.set_validity_zero_copy(mask, offset, len, None) }
    }

    /// Like [`set_validity`](Self::set_validity), but attempts a zero-copy path when
    /// `zero_copy` is provided and the offset is u64-aligned.
    ///
    /// Returns true if all values are null (caller can skip data export).
    pub(super) unsafe fn set_validity_zero_copy(
        &mut self,
        mask: &Mask,
        offset: usize,
        len: usize,
        zero_copy: Option<&ValidityData>,
    ) -> bool {
        match mask {
            Mask::AllTrue(_) => {
                self.set_all_true_validity();
                false
            }
            Mask::AllFalse(_) => {
                self.set_all_false_validity();
                true
            }
            Mask::Values(arr) => {
                let true_count = arr.bit_buffer().slice(offset..(offset + len)).true_count();
                if true_count == len {
                    self.set_all_true_validity()
                } else if true_count == 0 {
                    self.set_all_false_validity()
                } else if let Some(zc) = zero_copy.filter(|_| offset.is_multiple_of(64)) {
                    let u64_offset = offset / 64;
                    // SAFETY: the underlying buffer is u64-aligned (checked in
                    // can_zero_copy_validity) and the VectorBuffer keeps the data alive.
                    // data_ptr points into the buffer at the start of the validity bitmap.
                    unsafe { self.set_validity_data(u64_offset, len, zc) };
                } else {
                    // If zero_copy is available and offset is aligned, we should
                    // have taken the branch above. Assert this invariant.
                    assert!(
                        zero_copy.is_none() || !offset.is_multiple_of(64),
                        "zero-copy validity available and offset {offset} is aligned \
                         but copy path was taken"
                    );
                    let source = arr.bit_buffer().inner().as_slice();
                    copy_from_slice(
                        unsafe { self.ensure_validity_slice(len) },
                        source,
                        offset,
                        len,
                    );
                }

                true_count == 0
            }
        }
    }

    pub fn set_all_true_validity(&mut self) {
        unsafe { duckdb_vx_vector_set_all_valid(self.as_ptr()) };
    }

    pub fn set_all_false_validity(&mut self) {
        self.reference_value(&Value::null(&self.logical_type()));
    }
}
