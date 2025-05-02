use itertools::Itertools;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::arrays::BoolArray;
use crate::patches::Patches;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ToCanonical};

impl BoolArray {
    pub fn patch(self, patches: &Patches) -> VortexResult<Self> {
        let len = self.len();
        let (_, offset, indices, values) = patches.clone().into_parts();
        let indices = indices.to_primitive()?;
        let values = values.to_bool()?;

        let patched_validity =
            self.validity()
                .clone()
                .patch(len, offset, &indices, values.validity())?;

        let (mut own_values, bit_offset) = self.into_boolean_builder();
        match_each_integer_ptype!(indices.ptype(), |$I| {
            for (idx, value) in indices
                .as_slice::<$I>()
                .iter()
                .zip_eq(values.boolean_buffer().iter())
            {
                own_values.set_bit(*idx as usize - offset + bit_offset, value);
            }
        });

        Ok(Self::new(
            own_values.finish().slice(bit_offset, len),
            patched_validity,
        ))
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;

    use crate::ToCanonical;
    use crate::array::Array;
    use crate::arrays::BoolArray;

    #[test]
    fn patch_sliced_bools() {
        let arr = BoolArray::from(BooleanBuffer::new_set(12));
        let sliced = arr.slice(4, 12).unwrap();
        let (values, offset) = sliced.to_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.len(), 12);
        assert_eq!(values.as_slice(), &[255, 15]);
    }

    #[test]
    fn patch_sliced_bools_offset() {
        let arr = BoolArray::from(BooleanBuffer::new_set(15));
        let sliced = arr.slice(4, 15).unwrap();
        let (values, offset) = sliced.to_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.as_slice(), &[255, 127]);
    }
}
