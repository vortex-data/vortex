// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::mask::Mask;
use vortex::mask::MaskValues;

use crate::cpp::duckdb_vx_vector_set_all_valid;
use crate::duckdb::Value;
use crate::duckdb::VectorRef;
use crate::exporter::copy_from_slice;

impl VectorRef {
    pub(super) unsafe fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool {
        match mask {
            Mask::AllTrue(_) => {
                self.set_all_true_validity();
                false
            }
            Mask::AllFalse(_) => {
                self.set_all_false_validity();
                true
            }
            Mask::Values(arr) => self.set_validity_with_array(arr, len, offset),
        }
    }

    fn set_validity_with_array(&mut self, arr: &MaskValues, len: usize, offset: usize) -> bool {
        let true_count = arr.true_count();
        if true_count == arr.len() {
            self.set_all_true_validity();
            return false;
        } else if true_count == 0 {
            self.set_all_false_validity();
            return true;
        }

        let dest = unsafe { self.ensure_validity_slice(len) };
        let source = arr.bit_buffer().inner().as_slice();
        let ones = copy_from_slice(dest, source, offset, len);
        if ones == 0 {
            self.set_all_false_validity();
            true
        } else if ones == len {
            self.set_all_true_validity();
            false
        } else {
            false
        }
    }

    pub fn set_all_true_validity(&mut self) {
        unsafe { duckdb_vx_vector_set_all_valid(self.as_ptr()) };
    }

    pub fn set_all_false_validity(&mut self) {
        self.reference_value(&Value::null(&self.logical_type()));
    }
}
