// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::BoolArray;

impl BoolArray {
    pub fn opt_bool_vec(&self) -> VortexResult<Vec<Option<bool>>> {
        Ok(self
            .validity_mask()?
            .to_boolean_buffer()
            .into_iter()
            .zip(self.boolean_buffer().iter())
            .map(move |(valid, value)| valid.then_some(value))
            .collect_vec())
    }

    pub fn bool_vec(&self) -> VortexResult<Vec<bool>> {
        self.validity_mask()?
            .to_boolean_buffer()
            .into_iter()
            .zip(self.boolean_buffer().iter())
            .map(move |(valid, value)| {
                if !valid {
                    vortex_bail!("trying to get bool values from an array with null elements")
                }

                Ok(value)
            })
            .try_collect()
    }
}
