// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::Not;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::zip::ZipKernel;
use crate::validity::Validity;

impl ZipKernel for Struct {
    fn zip(
        if_true: ArrayView<'_, Struct>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<Struct>() else {
            return Ok(None);
        };
        assert_eq!(
            if_true.names(),
            if_false.names(),
            "input arrays to zip must have the same field names",
        );

        let fields = if_true
            .iter_unmasked_fields()
            .zip(if_false.iter_unmasked_fields())
            .map(|(t, f)| ArrayBuiltins::zip(mask, t.clone(), f.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        let v1 = if_true.validity();
        let v2 = if_false.validity();
        let validity = match (&v1, &v2) {
            (Validity::NonNullable, Validity::NonNullable) => Validity::NonNullable,
            (Validity::AllValid, Validity::AllValid) => Validity::AllValid,
            (Validity::AllInvalid, Validity::AllInvalid) => Validity::AllInvalid,

            (v1, v2) => {
                let mask_mask = mask.try_to_mask_fill_null_false(ctx)?;
                let v1m = v1.execute_mask(if_true.len(), ctx)?;
                let v2m = v2.execute_mask(if_false.len(), ctx)?;

                let combined = (v1m.bitand(&mask_mask)).bitor(&v2m.bitand(&mask_mask.not()));
                Validity::from_mask(
                    combined,
                    if_true.dtype.nullability() | if_false.dtype.nullability(),
                )
            }
        };

        Ok(Some(
            StructArray::try_new(if_true.names().clone(), fields, if_true.len(), validity)?
                .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::FieldNames;
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

        let result = mask.into_array().zip(if_true, if_false).unwrap();

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

        let result = mask.into_array().zip(if_true, if_false).unwrap();

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
