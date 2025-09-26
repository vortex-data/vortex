// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{DecimalDType, IntegerPType, match_each_integer_ptype};
use vortex_error::{VortexExpect as _, vortex_panic};
use vortex_scalar::{BigCast, NativeDecimalType, match_each_decimal_value_type};

use super::{DecimalArray, compatible_storage_type};
use crate::ToCanonical as _;
use crate::patches::Patches;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl DecimalArray {
    #[allow(clippy::cognitive_complexity)]
    pub fn patch(self, patches: &Patches) -> Self {
        let offset = patches.offset();
        let patch_indices = patches.indices().to_primitive();
        let patch_values = patches.values().to_decimal();

        let patched_validity = self.validity().clone().patch(
            self.len(),
            offset,
            patch_indices.as_ref(),
            patch_values.validity(),
        );
        assert_eq!(self.decimal_dtype(), patch_values.decimal_dtype());

        match_each_integer_ptype!(patch_indices.ptype(), |I| {
            let patch_indices = patch_indices.as_slice::<I>();
            match_each_decimal_value_type!(patch_values.values_type(), |PatchDVT| {
                let patch_values = patch_values.buffer::<PatchDVT>();
                match_each_decimal_value_type!(self.values_type(), |ValuesDVT| {
                    let buffer = self.buffer::<ValuesDVT>().into_mut();
                    patch_typed(
                        buffer,
                        self.decimal_dtype(),
                        patch_indices,
                        offset,
                        patch_values,
                        patched_validity,
                    )
                })
            })
        })
    }
}

fn patch_typed<I, ValuesDVT, PatchDVT>(
    mut buffer: BufferMut<ValuesDVT>,
    decimal_dtype: DecimalDType,
    patch_indices: &[I],
    patch_indices_offset: usize,
    patch_values: Buffer<PatchDVT>,
    patched_validity: Validity,
) -> DecimalArray
where
    I: IntegerPType,
    PatchDVT: NativeDecimalType,
    ValuesDVT: NativeDecimalType,
{
    if !compatible_storage_type(ValuesDVT::VALUES_TYPE, decimal_dtype) {
        vortex_panic!(
            "patch_typed: {:?} cannot represent every value in {}.",
            ValuesDVT::VALUES_TYPE,
            decimal_dtype
        )
    }

    for (idx, value) in patch_indices.iter().zip_eq(patch_values.into_iter()) {
        buffer[idx.as_() - patch_indices_offset] = <ValuesDVT as BigCast>::from(value).vortex_expect(
            "values of a given DecimalDType are representable in all compatible NativeDecimalType",
        );
    }

    DecimalArray::new(buffer.freeze(), decimal_dtype, patched_validity)
}
