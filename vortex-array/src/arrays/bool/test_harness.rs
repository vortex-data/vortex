// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use vortex_error::vortex_panic;

use crate::arrays::BoolArray;

impl BoolArray {
    pub fn opt_bool_vec(&self) -> Vec<Option<bool>> {
        self.validity_mask()
            .unwrap()
            .to_bit_buffer()
            .iter()
            .zip(self.bit_buffer().iter())
            .map(|(valid, value)| valid.then_some(value))
            .collect()
    }

    pub fn bool_vec(&self) -> Vec<bool> {
        self.validity_mask()
            .unwrap()
            .to_bit_buffer()
            .iter()
            .zip(self.bit_buffer().iter())
            .map(|(valid, value)| {
                if !valid {
                    vortex_panic!("trying to get bool values from an array with null elements")
                }

                value
            })
            .collect()
    }
}
