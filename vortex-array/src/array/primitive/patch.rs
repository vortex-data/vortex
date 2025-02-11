use arrow_buffer::ArrowNativeType;
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype, NativePType};
use vortex_error::VortexResult;

use crate::array::PrimitiveArray;
use crate::patches::Patches;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::IntoArrayVariant;

impl PrimitiveArray {
    #[allow(clippy::cognitive_complexity)]
    pub fn patch(self, patches: Patches) -> VortexResult<Self> {
        let (_, offset, patch_indices, patch_values) = patches.into_parts();
        let patch_indices = patch_indices.into_primitive()?;
        let patch_values = patch_values.into_primitive()?;

        let patched_validity = self.validity().patch(
            self.len(),
            offset,
            patch_indices.as_ref(),
            patch_values.validity(),
        )?;

        match_each_integer_ptype!(patch_indices.ptype(), |$I| {
            match_each_native_ptype!(self.ptype(), |$T| {
                self.patch_typed::<$T, $I>(patch_indices, offset, patch_values, patched_validity)
            })
        })
    }

    fn patch_typed<T, I>(
        self,
        patch_indices: PrimitiveArray,
        patch_indices_offset: usize,
        patch_values: PrimitiveArray,
        patched_validity: Validity,
    ) -> VortexResult<Self>
    where
        T: NativePType + ArrowNativeType,
        I: NativePType + ArrowNativeType,
    {
        let mut own_values = self.into_buffer_mut::<T>();

        let patch_indices = patch_indices.as_slice::<I>();
        let patch_values = patch_values.as_slice::<T>();
        for (idx, value) in itertools::zip_eq(patch_indices, patch_values) {
            own_values[idx.as_usize() - patch_indices_offset] = *value;
        }
        Ok(Self::new(own_values, patched_validity))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::compute::slice;
    use crate::validity::Validity;
    use crate::IntoArrayVariant;

    #[test]
    fn patch_sliced() {
        let input = PrimitiveArray::new(buffer![2u32; 10], Validity::AllValid);
        let sliced = slice(input, 2, 8).unwrap();
        assert_eq!(
            sliced.into_primitive().unwrap().as_slice::<u32>(),
            &[2u32; 6]
        );
    }
}
