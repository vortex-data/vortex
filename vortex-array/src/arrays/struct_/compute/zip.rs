// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{BitAnd, BitOr, Not};

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{StructArray, StructVTable};
use crate::compute::{ZipKernel, ZipKernelAdapter, zip};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, register_kernel};

impl ZipKernel for StructVTable {
    fn zip(
        &self,
        if_true: &StructArray,
        if_false: &dyn Array,
        mask: &Mask,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<StructVTable>() else {
            return Ok(None);
        };
        assert_eq!(
            if_true.len(),
            if_false.len(),
            "ComputeFn::invoke checks that arrays have the same size"
        );
        assert_eq!(
            if_true.names(),
            if_false.names(),
            "Zip checks that arrays type"
        );

        let fields = if_true
            .fields()
            .iter()
            .zip(if_false.fields().iter())
            .map(|(t, f)| zip(t, f, mask))
            .collect::<VortexResult<Vec<_>>>()?;

        let validity = match (if_true.validity(), if_false.validity()) {
            (&Validity::NonNullable, &Validity::NonNullable) => Validity::NonNullable,
            (&Validity::AllValid, &Validity::AllValid) => Validity::AllValid,
            (&Validity::AllInvalid, &Validity::AllInvalid) => Validity::AllInvalid,

            (v1, v2) => {
                let v1m = v1.to_mask(if_true.len());
                let v2m = v2.to_mask(if_false.len());

                let combined = (v1m.bitand(mask)).bitor(&v2m.bitand(&mask.not()));
                Validity::from_mask(
                    combined,
                    if_true.dtype.nullability() | if_false.dtype.nullability(),
                )
            }
        };

        Ok(Some(
            StructArray::try_new(if_true.names().clone(), fields, if_true.len(), validity)?
                .to_array(),
        ))
    }
}

register_kernel!(ZipKernelAdapter(StructVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability};
    use vortex_mask::Mask;

    use crate::arrays::{PrimitiveArray, StructArray};
    use crate::canonical::ToCanonical;
    use crate::compute::zip;
    use crate::validity::Validity;
    use crate::{Array, IntoArray};

    #[test]
    fn test_validity_zip_both_validity_array() {
        // Both structs have Validity::Array
        let if_true = StructArray::try_from_iter([(
            "field",
            PrimitiveArray::from_option_iter([Some(1), None, Some(3), None]).into_array(),
        )])
        .unwrap()
        .into_array();

        let if_false = StructArray::try_from_iter([(
            "field",
            PrimitiveArray::from_option_iter([None, Some(20), None, Some(40)]).into_array(),
        )])
        .unwrap()
        .into_array();

        let mask = Mask::from_iter([false, false, true, false]);

        let result = zip(&if_true, &if_false, &mask).unwrap();

        insta::assert_snapshot!(result.display_table(), @r"
        ┌───────┐
        │ field │
        ├───────┤
        │ null  │
        ├───────┤
        │ 20i32 │
        ├───────┤
        │ 3i32  │
        ├───────┤
        │ 40i32 │
        └───────┘
        ");
    }

    #[test]
    fn test_validity_zip_allvalid_and_array() {
        // One struct is AllValid, the other has Validity::Array
        let if_true = StructArray::try_from_iter([(
            "field",
            PrimitiveArray::from_option_iter([Some(10), Some(20), Some(30), Some(40)]).into_array(),
        )])
        .unwrap()
        .into_array();

        let if_false = StructArray::try_from_iter([(
            "field",
            PrimitiveArray::from_option_iter([Some(1), None, Some(3), Some(4)]).into_array(),
        )])
        .unwrap()
        .into_array();

        let mask = Mask::from_iter([true, false, false, false]);

        let result = zip(&if_true, &if_false, &mask).unwrap();

        insta::assert_snapshot!(result.display_table(), @r"
        ┌───────┐
        │ field │
        ├───────┤
        │ 10i32 │
        ├───────┤
        │ null  │
        ├───────┤
        │ 3i32  │
        ├───────┤
        │ 4i32  │
        └───────┘
        ");
    }
}
