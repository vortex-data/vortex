// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::zip::ZipKernel;
use crate::scalar_fn::fns::zip::zip_validity;

/// Zips two fixed-size list arrays by zipping their elements with a mask that repeats each row's
/// bit `list_size` times.
impl ZipKernel for FixedSizeList {
    fn zip(
        if_true: ArrayView<'_, FixedSizeList>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<FixedSizeList>() else {
            return Ok(None);
        };

        let list_size = if_true.list_size();
        vortex_ensure_eq!(list_size, if_false.list_size());

        // Null mask entries select `if_false`, matching `Zip`'s SQL ELSE semantics.
        let mask = mask.clone().fill_null(false)?.execute::<Mask>(ctx)?;
        let Mask::Values(mask_values) = &mask else {
            // Defer trivial masks to the generic zip, which just casts the surviving side.
            return Ok(None);
        };

        let elements = expand_mask(mask_values, list_size)
            .into_array()
            .zip(if_true.elements().clone(), if_false.elements().clone())?;
        let validity = zip_validity(if_true.validity()?, if_false.validity()?, &mask)?;

        Ok(Some(
            FixedSizeListArray::try_new(elements, list_size, validity, if_true.len())?.into_array(),
        ))
    }
}

/// Repeats each row's mask bit `list_size` times, so the result selects whole lists of elements.
fn expand_mask(mask: &MaskValues, list_size: u32) -> Mask {
    let list_size = list_size as usize;
    if list_size == 0 {
        return Mask::new_false(0);
    }

    let slices = mask
        .slices()
        .iter()
        .map(|&(start, end)| (start * list_size, end * list_size))
        .collect();
    Mask::from_slices(mask.len() * list_size, slices)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::FixedSizeList;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity_in;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::memory::BufferAllocatorRef;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn array_from_scalars(dtype: &DType, scalars: &[Scalar]) -> VortexResult<ArrayRef> {
        let mut builder =
            builder_with_capacity_in(dtype, scalars.len(), BufferAllocatorRef::static_ref());
        for scalar in scalars {
            builder.append_scalar(scalar)?;
        }
        Ok(builder.finish())
    }

    fn i32_element() -> Arc<DType> {
        Arc::new(DType::Primitive(PType::I32, Nullability::Nullable))
    }

    fn struct_element() -> Arc<DType> {
        Arc::new(DType::struct_(
            [("x", DType::Primitive(PType::I32, Nullability::NonNullable))],
            Nullability::NonNullable,
        ))
    }

    /// Element `j` of list `i` is `offset + 2 * i + j`. The first element of every fifth list is
    /// null when the element type is nullable, and every seventh list is null when the list type
    /// is nullable.
    fn list(element_dtype: &Arc<DType>, offset: i32, i: i32, nullability: Nullability) -> Scalar {
        let dtype = DType::FixedSizeList(Arc::clone(element_dtype), 2, nullability);
        if nullability.is_nullable() && i % 7 == 0 {
            return Scalar::null(dtype);
        }

        let elements = (0..2)
            .map(|j| {
                let value = offset + 2 * i + j;
                match element_dtype.as_ref() {
                    DType::Struct(..) => {
                        Scalar::struct_(element_dtype.as_ref().clone(), [Scalar::from(value)])
                    }
                    _ if j == 0 && i % 5 == 0 => Scalar::null(element_dtype.as_ref().clone()),
                    _ => Scalar::primitive(value, Nullability::Nullable),
                }
            })
            .collect();
        Scalar::fixed_size_list(Arc::clone(element_dtype), elements, nullability)
    }

    /// A fragmented mask that spans more than 64 rows selects whole lists, their element values
    /// and their validity from the chosen side.
    #[rstest]
    #[case::non_nullable(i32_element(), Nullability::NonNullable, Nullability::NonNullable)]
    #[case::nullable_if_true(i32_element(), Nullability::Nullable, Nullability::NonNullable)]
    #[case::nullable_if_false(i32_element(), Nullability::NonNullable, Nullability::Nullable)]
    #[case::nested_elements(struct_element(), Nullability::Nullable, Nullability::Nullable)]
    fn zip_fixed_size_lists(
        #[case] element_dtype: Arc<DType>,
        #[case] true_nullability: Nullability,
        #[case] false_nullability: Nullability,
    ) -> VortexResult<()> {
        let len = 70;
        let selected = |i: i32| i % 3 == 0 || i == 64;
        let lists = |offset: i32, nullability: Nullability| {
            (0..len)
                .map(|i| list(&element_dtype, offset, i, nullability))
                .collect::<Vec<_>>()
        };
        let nullability = true_nullability | false_nullability;
        let true_dtype = DType::FixedSizeList(Arc::clone(&element_dtype), 2, true_nullability);
        let false_dtype = DType::FixedSizeList(Arc::clone(&element_dtype), 2, false_nullability);
        let result_dtype = DType::FixedSizeList(Arc::clone(&element_dtype), 2, nullability);

        let if_true = array_from_scalars(&true_dtype, &lists(0, true_nullability))?;
        let if_false = array_from_scalars(&false_dtype, &lists(1_000, false_nullability))?;
        let mask = Mask::from_iter((0..len).map(selected));

        let mut ctx = SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<FixedSizeList>());

        let expected = (0..len)
            .map(|i| {
                let selected_list = if selected(i) {
                    list(&element_dtype, 0, i, true_nullability)
                } else {
                    list(&element_dtype, 1_000, i, false_nullability)
                };
                selected_list.cast(&result_dtype)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        assert_arrays_eq!(
            result,
            array_from_scalars(&result_dtype, &expected)?,
            &mut ctx
        );
        Ok(())
    }
}
