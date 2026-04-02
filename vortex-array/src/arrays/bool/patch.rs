// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;

impl BoolArray {
    pub fn patch(self, patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let len = self.len();
        let offset = patches.offset();
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = patches.values().clone().execute::<BoolArray>(ctx)?;

        let patched_validity =
            self.validity()
                .patch(len, offset, patches.indices(), &values.validity(), ctx)?;

        let bit_buffer = self.into_bit_buffer();
        let mut own_values = bit_buffer
            .try_into_mut()
            .unwrap_or_else(|bb| BitBufferMut::copy_from(&bb));
        match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            for (idx, value) in indices
                .as_slice::<I>()
                .iter()
                .zip_eq(values.to_bit_buffer().iter())
            {
                #[allow(clippy::cast_possible_truncation)]
                own_values.set_to(*idx as usize - offset, value);
            }
        });

        Ok(Self::new(own_values.freeze(), patched_validity))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;

    #[test]
    fn patch_sliced_bools() {
        let arr = BoolArray::from(BitBuffer::new_set(12));
        let sliced = arr.into_array().slice(4..12).unwrap();
        let expected = BoolArray::from_iter([true; 8]);
        assert_arrays_eq!(sliced, expected);
    }

    #[test]
    fn patch_sliced_bools_offset() {
        let arr = BoolArray::from(BitBuffer::new_set(15));
        let sliced = arr.into_array().slice(4..15).unwrap();
        let expected = BoolArray::from_iter([true; 11]);
        assert_arrays_eq!(sliced, expected);
    }
}
