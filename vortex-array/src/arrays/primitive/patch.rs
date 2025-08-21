// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_dtype::{NativePType, match_each_integer_ptype, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::ToCanonical;
use crate::arrays::PrimitiveArray;
use crate::patches::Patches;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl PrimitiveArray {
    #[allow(clippy::cognitive_complexity)]
    pub fn patch(self, patches: &Patches) -> VortexResult<Self> {
        let patch_indices = patches.indices().to_primitive()?;
        let patch_values = patches.values().to_primitive()?;

        let patched_validity = self.validity().clone().patch(
            self.len(),
            patches.offset(),
            patch_indices.as_ref(),
            patch_values.validity(),
        )?;
        match_each_integer_ptype!(patch_indices.ptype(), |I| {
            match_each_native_ptype!(self.ptype(), |T| {
                self.patch_typed::<T, I>(
                    patch_indices,
                    patches.offset(),
                    patch_values,
                    patched_validity,
                )
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
        T: NativePType,
        I: NativePType + AsPrimitive<usize>,
    {
        let mut own_values = self.into_buffer_mut::<T>();

        let patch_indices = patch_indices.as_slice::<I>();
        let patch_values = patch_values.as_slice::<T>();
        for (idx, value) in itertools::zip_eq(patch_indices, patch_values) {
            own_values[idx.as_() - patch_indices_offset] = *value;
        }
        Ok(Self::new(own_values, patched_validity))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::ToCanonical;
    use crate::validity::Validity;

    #[test]
    fn patch_sliced() {
        let input = PrimitiveArray::new(buffer![2u32; 10], Validity::AllValid);
        let sliced = input.slice(2, 8);
        assert_eq!(sliced.to_primitive().unwrap().as_slice::<u32>(), &[2u32; 6]);
    }
}
