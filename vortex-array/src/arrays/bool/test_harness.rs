// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;

impl BoolArray {
    pub fn opt_bool_vec(&self) -> Vec<Option<bool>> {
        self.validity()
            .vortex_expect("failed to get validity")
            .to_mask(
                self.as_ref().len(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .vortex_expect("Failed to compute validity mask")
            .to_bit_buffer()
            .iter()
            .zip(self.to_bit_buffer().iter())
            .map(|(valid, value)| valid.then_some(value))
            .collect()
    }

    pub fn bool_vec(&self) -> Vec<bool> {
        self.validity()
            .vortex_expect("failed to get validity")
            .to_mask(
                self.as_ref().len(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )
            .vortex_expect("Failed to compute validity mask")
            .to_bit_buffer()
            .iter()
            .zip(self.to_bit_buffer().iter())
            .map(|(valid, value)| {
                if !valid {
                    vortex_panic!("trying to get bool values from an array with null elements")
                }

                value
            })
            .collect()
    }
}
