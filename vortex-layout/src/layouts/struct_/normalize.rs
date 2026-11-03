// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// Remove nulls hidden by ancestor structs without changing the enclosing struct's values.
/// Rebuilding arrays also discards child statistics that the newly valid rows would invalidate.
pub(super) fn normalize_struct_children(
    array: ArrayRef,
    visible: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if array.dtype().is_struct() {
        let array = array.execute::<StructArray>(ctx)?;
        let validity = array.as_ref().validity()?.execute_mask(array.len(), ctx)?;
        let child_visible = visible & &validity;
        let fields = array
            .iter_unmasked_fields()
            .map(|field| normalize_struct_children(field.clone(), &child_visible, ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = Validity::from_mask(validity | &!visible, array.dtype().nullability());
        return Ok(
            StructArray::try_new(array.names().clone(), fields, array.len(), validity)?.into_array(),
        );
    }

    if visible.all_true() || !array.dtype().is_nullable() {
        return Ok(array);
    }

    // These types have non-null defaults. Union and variant children are left unchanged rather
    // than choosing a potentially expensive or unavailable default physical representation.
    if !matches!(
        array.dtype(),
        DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(_)
            | DType::Binary(_)
            | DType::List(..)
            | DType::FixedSizeList(..)
    ) {
        return Ok(array);
    }

    // Null Arrow values can contain arbitrary views or list offsets. Select a real default
    // value on hidden rows instead of making that physical data observable to compression.
    let default = Scalar::default_value(&array.dtype().with_nullability(Nullability::NonNullable))
        .cast(array.dtype())?;
    let defaults = ConstantArray::new(default, array.len()).into_array();
    Ok(visible
        .clone()
        .into_array()
        .zip(array, defaults)?
        .execute::<Canonical>(ctx)?
        .into_array())
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::normalize_struct_children;
    use crate::test::SESSION;

    #[test]
    fn preserves_child_nulls_and_original_ancestor_visibility() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let child = PrimitiveArray::new(
            buffer![1i32, i32::MAX, 3, 99, 5, 77],
            Validity::from(BitBuffer::from_iter([true, false, false, true, true, false])),
        );
        let inner = StructArray::try_from_iter_with_validity(
            [("value", child.into_array())],
            Validity::from(BitBuffer::from_iter([true, false, true, true, true, false])),
        )?;
        let input = StructArray::try_from_iter_with_validity(
            [("inner", inner.into_array())],
            Validity::from(BitBuffer::from_iter([true, false, true, false, true, false])),
        )?
        .into_array();
        let normalized = normalize_struct_children(input.clone(), &Mask::new_true(6), &mut ctx)?;
        assert_arrays_eq!(&normalized, &input, &mut ctx);

        let outer = normalized.execute::<StructArray>(&mut ctx)?;
        let inner = outer
            .unmasked_field(0)
            .clone()
            .execute::<StructArray>(&mut ctx)?;
        assert!(inner.as_ref().validity()?.execute_mask(6, &mut ctx)?.all_true());
        let expected = PrimitiveArray::from_option_iter([
            Some(1i32),
            Some(0),
            None,
            Some(0),
            Some(5),
            Some(0),
        ])
        .into_array();
        assert_arrays_eq!(inner.unmasked_field(0), &expected, &mut ctx);
        Ok(())
    }
}
