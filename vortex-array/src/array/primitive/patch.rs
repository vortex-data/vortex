use itertools::Itertools;
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype};
use vortex_error::{vortex_err, VortexExpect as _, VortexResult};

use crate::array::PrimitiveArray;
use crate::patches::Patches;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayLen, IntoArrayVariant};

impl PrimitiveArray {
    #[allow(clippy::cognitive_complexity)]
    pub fn patch(self, patches: Patches) -> VortexResult<Self> {
        let (_, indices, patch_values) = patches.into_parts();
        let indices = indices.into_primitive()?;
        let patch_values = patch_values.into_primitive()?;

        let patched_validity =
            self.validity()
                .patch(self.len(), indices.as_ref(), patch_values.validity())?;

        match_each_integer_ptype!(indices.ptype(), |$I| {
            match_each_native_ptype!(self.ptype(), |$T| {
                let mut own_values = self.into_maybe_null_vec::<$T>();
                let indices_slice = indices
                    .try_into_maybe_null_vec::<$I>()
                    .map_err(|_| vortex_err!("could not zero copy patched left_parts_decoded"))
                    .vortex_expect("there are no aliases to this primitive array");
                let patch_values_slice = patch_values
                    .try_into_maybe_null_vec::<$T>()
                    .map_err(|_| vortex_err!("could not zero copy patched left_parts_decoded"))
                    .vortex_expect("there are no aliases to this primitive array");
                for (idx, value) in indices_slice.into_iter().zip_eq(patch_values_slice.into_iter()) {
                    own_values[idx as usize] = value;
                }
                Ok(Self::from_vec(own_values, patched_validity))
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::slice;
    use crate::validity::Validity;
    use crate::IntoArrayVariant;

    #[test]
    fn patch_sliced() {
        let input = PrimitiveArray::from_vec(vec![2u32; 10], Validity::AllValid);
        let sliced = slice(input, 2, 8).unwrap();
        assert_eq!(
            sliced
                .into_primitive()
                .unwrap()
                .into_maybe_null_vec::<u32>(),
            vec![2u32; 6]
        );
    }
}
