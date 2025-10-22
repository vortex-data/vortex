// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_dtype::match_each_unsigned_integer_ptype;

use crate::ToCanonical;
use crate::arrays::BoolArray;
use crate::patches::Patches;
use crate::vtable::ValidityHelper;

impl BoolArray {
    pub fn patch(self, patches: &Patches) -> Self {
        let len = self.len();
        let offset = patches.offset();
        let indices = patches.indices().to_primitive();
        let values = patches.values().to_bool();

        let patched_validity =
            self.validity()
                .clone()
                .patch(len, offset, indices.as_ref(), values.validity());

        let mut own_values = self.into_bit_buffer().into_mut();
        match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            for (idx, value) in indices
                .as_slice::<I>()
                .iter()
                .zip_eq(values.bit_buffer().iter())
            {
                #[allow(clippy::cast_possible_truncation)]
                own_values.set_to(*idx as usize - offset, value);
            }
        });

        Self::from_bit_buffer(own_values.freeze(), patched_validity)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;

    use crate::ToCanonical;
    use crate::arrays::BoolArray;

    #[test]
    fn patch_sliced_bools() {
        let arr = BoolArray::from(BitBuffer::new_set(12));
        let sliced = arr.slice(4..12);
        let values = sliced.to_bool().into_bit_buffer().into_mut();
        assert_eq!(values.len(), 8);
        assert_eq!(values.as_slice(), &[255, 255]);
    }

    #[test]
    fn patch_sliced_bools_offset() {
        let arr = BoolArray::from(BitBuffer::new_set(15));
        let sliced = arr.slice(4..15);
        let values = sliced.to_bool().into_bit_buffer().into_mut();
        assert_eq!(values.len(), 11);
        assert_eq!(values.as_slice(), &[255, 255]);
    }
}
