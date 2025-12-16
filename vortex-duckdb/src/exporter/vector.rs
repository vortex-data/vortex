// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::mask::Mask;

use crate::Value;
use crate::duckdb::Vector;
use crate::exporter::copy_from_slice;

impl Vector {
    pub(super) unsafe fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool {
        match mask {
            Mask::AllTrue(_) => {
                // We only need to blank out validity if there is already a slice allocated.
                // SAFETY: Caller guarantees this.
                unsafe { self.set_all_true_validity(len) }
                false
            }
            Mask::AllFalse(_) => {
                // SAFETY: Caller guarantees this.
                self.set_all_false_validity();
                true
            }
            Mask::Values(arr) => {
                let true_count = arr.bit_buffer().true_count();
                if true_count == len {
                    unsafe { self.set_all_true_validity(len) }
                } else if true_count == 0 {
                    self.set_all_false_validity()
                } else {
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

    pub(super) unsafe fn set_all_true_validity(&mut self, len: usize) {
        if let Some(validity) = unsafe { self.validity_bitslice_mut(len) } {
            validity.fill(true);
        }
    }

    pub(super) fn set_all_false_validity(&mut self) {
        self.reference_value(&Value::null(&self.logical_type()));
    }
}
