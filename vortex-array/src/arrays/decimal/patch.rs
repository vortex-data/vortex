use arrow_buffer::ArrowNativeType;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_scalar::{NativeDecimalType, match_each_decimal_value_type};

use super::DecimalArray;
use crate::ToCanonical as _;
use crate::arrays::PrimitiveArray;
use crate::patches::Patches;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl DecimalArray {
    pub fn patch(self, patches: &Patches) -> VortexResult<Self> {
        let offset = patches.offset();
        let patch_indices = patches.indices().to_primitive()?;
        let patch_values = patches.values().to_decimal()?;

        let patched_validity = self.validity().clone().patch(
            self.len(),
            offset,
            patch_indices.as_ref(),
            patch_values.validity(),
        )?;
        assert_eq!(self.decimal_dtype(), patch_values.decimal_dtype());
        assert_eq!(self.values_type(), patch_values.values_type());
        match_each_integer_ptype!(patch_indices.ptype(), |I| {
            match_each_decimal_value_type!(self.values_type(), |T| {
                self.patch_typed::<T, I>(patch_indices, offset, patch_values, patched_validity)
            })
        })
    }

    fn patch_typed<T, I>(
        self,
        patch_indices: PrimitiveArray,
        patch_indices_offset: usize,
        patch_values: DecimalArray,
        patched_validity: Validity,
    ) -> VortexResult<Self>
    where
        T: NativeDecimalType,
        I: NativePType + ArrowNativeType,
    {
        let mut own_values = self.buffer::<T>().into_mut();

        let patch_indices = patch_indices.as_slice::<I>();
        let patch_values = patch_values.buffer::<T>();
        for (idx, value) in itertools::zip_eq(patch_indices, patch_values) {
            own_values[idx.as_usize() - patch_indices_offset] = value;
        }
        Ok(Self::new(
            own_values.freeze(),
            self.decimal_dtype(),
            patched_validity,
        ))
    }
}
