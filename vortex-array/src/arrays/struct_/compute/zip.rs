// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::Not;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::compute::ZipKernel;
use crate::compute::ZipKernelAdapter;
use crate::compute::zip;
use crate::register_kernel;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

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
            if_true.names(),
            if_false.names(),
            "input arrays to zip must have the same field names",
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
    use vortex_dtype::FieldNames;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::compute::zip;
    use crate::validity::Validity;

    #[test]
    fn test_validity_zip_both_validity_array() {
        // Both structs have Validity::Array
        let if_true = StructArray::new(
            FieldNames::from_iter(["field"]),
            vec![PrimitiveArray::from_iter([1, 2, 3, 4]).into_array()],
            4,
            Validity::from_iter([true, false, true, false]),
        )
        .into_array();

        let if_false = StructArray::new(
            FieldNames::from_iter(["field"]),
            vec![PrimitiveArray::from_iter([10, 20, 30, 40]).into_array()],
            4,
            Validity::from_iter([false, true, false, true]),
        )
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
        let if_true = StructArray::new(
            FieldNames::from_iter(["a"]),
            vec![PrimitiveArray::from_iter([1, 2, 3, 4]).into_array()],
            4,
            Validity::AllValid,
        )
        .into_array();

        let if_false = StructArray::new(
            FieldNames::from_iter(["a"]),
            vec![PrimitiveArray::from_iter([10, 20, 30, 40]).into_array()],
            4,
            Validity::from_iter([false, false, true, true]),
        )
        .into_array();

        let mask = Mask::from_iter([true, false, false, false]);

        let result = zip(&if_true, &if_false, &mask).unwrap();

        insta::assert_snapshot!(result.display_table(), @r"
        ┌───────┐
        │   a   │
        ├───────┤
        │ 1i32  │
        ├───────┤
        │ null  │
        ├───────┤
        │ 30i32 │
        ├───────┤
        │ 40i32 │
        └───────┘
        ");
    }
}
