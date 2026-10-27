// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;

impl BoolArray {
    pub fn patch(self, patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let len = self.len();
        let offset = patches.offset();
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = patches.values().clone().execute::<BoolArray>(ctx)?;

        let patched_validity =
            self.validity()?
                .patch(len, offset, patches.indices(), &values.validity()?, ctx)?;

        let bit_buffer = self.into_bit_buffer();
        let mut own_values = bit_buffer
            .try_into_mut()
            .unwrap_or_else(|bb| BitBufferMut::copy_from(&bb));
        match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            for (idx, value) in indices
                .as_slice::<I>()
                .iter()
                .zip_eq(values.bit_buffer_view().iter())
            {
                #[allow(clippy::cast_possible_truncation)]
                own_values.set_to(*idx as usize - offset, value);
            }
        });

        Ok(Self::new(own_values.freeze(), patched_validity))
    }
}
