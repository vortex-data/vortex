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
        let indices = patches.indices().clone().into_primitive()?;
        let values = patches.values().clone().into_primitive()?;

        let patched_validity =
            self.validity()
                .patch(self.len(), indices.as_ref(), values.validity())?;

        match_each_integer_ptype!(indices.ptype(), |$I| {
            match_each_native_ptype!(self.ptype(), |$T| {
                let mut own_values = self.into_maybe_null_slice::<$T>();
                for (idx, value) in indices.into_maybe_null_slice::<$I>().into_iter().zip_eq(values.into_maybe_null_slice::<$T>().into_iter()) {
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
                .into_maybe_null_slice::<u32>(),
            vec![2u32; 6]
        );
    }
}
