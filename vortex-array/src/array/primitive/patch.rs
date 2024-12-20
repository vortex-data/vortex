use itertools::Itertools;
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::array::PrimitiveArray;
use crate::patches::Patches;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayLen, IntoArrayVariant};

impl PrimitiveArray {
    #[allow(clippy::cognitive_complexity)]
    pub fn patch(self, patches: Patches) -> VortexResult<Self> {
        let (_, indices, values) = patches.into_parts();
        let indices = indices.into_primitive()?;
        let values = values.into_primitive()?;

        let patched_validity =
            self.validity()
                .patch(self.len(), indices.as_ref(), values.validity())?;

        match_each_integer_ptype!(indices.ptype(), |$I| {
            match_each_native_ptype!(self.ptype(), |$T| {
                let mut own_values = self.into_buffer_mut::<$T>();
                for (idx, value) in indices.as_slice::<$I>().iter().zip_eq(values.as_slice::<$T>().iter()) {
                    own_values[*idx as usize] = *value;
                }
                Ok(Self::new(own_values.freeze(), patched_validity))
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
        let input = PrimitiveArray::copy_from_vec(vec![2u32; 10], Validity::AllValid);
        let sliced = slice(input, 2, 8).unwrap();
        assert_eq!(
            sliced.into_primitive().unwrap().as_slice::<u32>(),
            &[2u32; 6]
        );
    }
}
