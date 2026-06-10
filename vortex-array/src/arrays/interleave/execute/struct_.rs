// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Struct-value execution: the [`Interleave`](super::super::Interleave) path for struct values.
//!
//! A struct gather is a per-field gather plus a gather of the struct-level validity, so the
//! kernel stays lazy: it rewrites the interleave of structs into a struct of per-field
//! interleaves (sharing the selectors), and lets the scheduler drive each field's interleave
//! through its own value-typed kernel.

use vortex_error::VortexResult;

use super::super::Interleave;
use super::super::InterleaveArray;
use super::super::InterleaveArrayExt;
use crate::IntoArray;
use crate::array::Array;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayExt;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::require_child;

/// Rewrites an interleave of `N` struct values into a struct of per-field interleaves, routed by
/// the same `array_indices` / `row_indices` selectors.
pub(super) fn execute(
    array: Array<Interleave>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();

    // Drive every value to the canonical struct encoding so its fields can be unzipped. The
    // selectors stay as-is: they are cloned into each per-field interleave.
    let mut array = array;
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i => Struct);
    }

    let len = array.as_ref().len();
    // The output validity is the interleave of the values' validities, provided by the encoding's
    // validity implementation.
    let validity = array.as_ref().validity()?;

    let names = array.value(0).as_::<Struct>().names().clone();
    let fields = (0..names.len())
        .map(|field| {
            let field_values = (0..num_values)
                .map(|i| array.value(i).as_::<Struct>().unmasked_field(field).clone())
                .collect();
            InterleaveArray::try_new(
                field_values,
                array.array_indices().clone(),
                array.row_indices().clone(),
            )
            .map(IntoArray::into_array)
        })
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(ExecutionResult::done(StructArray::try_new(
        names, fields, len, validity,
    )?))
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::InterleaveArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::FieldNames;
    use crate::validity::Validity;

    fn selectors(indices: &[(u32, u32)]) -> (ArrayRef, ArrayRef) {
        (
            PrimitiveArray::from_iter(indices.iter().map(|&(a, _)| a)).into_array(),
            PrimitiveArray::from_iter(indices.iter().map(|&(_, r)| r)).into_array(),
        )
    }

    fn person(names: &[&str], flags: &[bool], validity: Validity) -> ArrayRef {
        StructArray::new(
            FieldNames::from_iter(["name", "flag"]),
            vec![
                VarBinViewArray::from_iter_str(names.iter().copied()).into_array(),
                BoolArray::from_iter(flags.iter().copied()).into_array(),
            ],
            names.len(),
            validity,
        )
        .into_array()
    }

    #[test]
    fn interleave_structs_reorders_and_repeats() -> VortexResult<()> {
        let v0 = person(
            &["alice", "a name long enough to be outlined"],
            &[true, false],
            Validity::NonNullable,
        );
        let v1 = person(&["bob"], &[false], Validity::NonNullable);
        let (array_indices, row_indices) = selectors(&[(1, 0), (0, 1), (0, 0), (0, 1)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        let expected = person(
            &[
                "bob",
                "a name long enough to be outlined",
                "alice",
                "a name long enough to be outlined",
            ],
            &[false, false, true, false],
            Validity::NonNullable,
        );
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_structs_unifies_field_nullability() -> VortexResult<()> {
        // v0's field is non-nullable while v1's is nullable (with a null); the interleave's field
        // dtype is the recursive nullability union.
        let v0 = StructArray::new(
            FieldNames::from_iter(["x"]),
            vec![BoolArray::from_iter([true, false]).into_array()],
            2,
            Validity::NonNullable,
        )
        .into_array();
        let v1 = StructArray::new(
            FieldNames::from_iter(["x"]),
            vec![BoolArray::from_iter([Some(true), None]).into_array()],
            2,
            Validity::NonNullable,
        )
        .into_array();
        let (array_indices, row_indices) = selectors(&[(1, 1), (0, 0), (0, 1)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        let expected = StructArray::new(
            FieldNames::from_iter(["x"]),
            vec![BoolArray::from_iter([None, Some(true), Some(false)]).into_array()],
            3,
            Validity::NonNullable,
        );
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_structs_with_null_rows() -> VortexResult<()> {
        let v0 = person(
            &["alice", "carol"],
            &[true, true],
            Validity::from_iter([true, false]),
        );
        let v1 = person(&["bob"], &[false], Validity::from_iter([false]));
        let (array_indices, row_indices) = selectors(&[(0, 1), (1, 0), (0, 0)]);

        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?;

        let expected = person(
            &["carol", "bob", "alice"],
            &[true, false, true],
            Validity::from_iter([false, false, true]),
        );
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }
}
