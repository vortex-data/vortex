use itertools::Itertools;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::array::BoolArray;
use crate::patches::Patches;
use crate::variants::PrimitiveArrayTrait;
use crate::IntoArrayVariant;

impl BoolArray {
    pub fn patch(self, patches: Patches) -> VortexResult<Self> {
        let length = self.len();
        let indices = patches.indices().clone().into_primitive()?;
        let values = patches.values().clone().into_bool()?;

        let patched_validity = self
            .validity()
            .patch(self.len(), &indices, values.validity())?;

        let (mut own_values, bit_offset) = self.into_boolean_builder();
        match_each_integer_ptype!(indices.ptype(), |$I| {
            for (idx, value) in indices
                .as_slice::<$I>()
                .iter()
                .zip_eq(values.boolean_buffer().iter())
            {
                own_values.set_bit(*idx as usize + bit_offset, value);
            }
        });

        Self::try_new(
            own_values.finish().slice(bit_offset, length),
            patched_validity,
        )
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;

    use crate::array::BoolArray;
    use crate::compute::slice;
    use crate::IntoArrayVariant;

    #[test]
    fn patch_sliced_bools() {
        let arr = BoolArray::from(BooleanBuffer::new_set(12));
        let sliced = slice(arr, 4, 12).unwrap();
        let (values, offset) = sliced.into_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.as_slice(), &[255, 15]);
    }

    #[test]
    fn patch_sliced_bools_offset() {
        let arr = BoolArray::from(BooleanBuffer::new_set(15));
        let sliced = slice(arr, 4, 15).unwrap();
        let (values, offset) = sliced.into_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 4);
        assert_eq!(values.as_slice(), &[255, 127]);
    }

    #[test]
    fn patch_sliced_bools_even() {
        let arr = BoolArray::from(BooleanBuffer::new_set(31));
        let sliced = slice(arr, 8, 24).unwrap();
        let (values, offset) = sliced.into_bool().unwrap().into_boolean_builder();
        assert_eq!(offset, 0);
        assert_eq!(values.as_slice(), &[255, 255]);
    }
}
