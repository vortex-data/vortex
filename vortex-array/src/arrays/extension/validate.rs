// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Value validation after the root array and its ancestor validity are known.

use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::arrays::Chunked;
use crate::arrays::Constant;
use crate::arrays::PrimitiveArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::listview::referenced_elements_mask;
use crate::arrays::map::MapArraySlotsExt;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::union::UnionArrayExt;
use crate::arrays::union::UnionArraySlotsExt;
use crate::dtype::DType;

/// Checks each extension value that is visible from the root of `array`.
///
/// A child value is visible when a valid row references it and every ancestor on that path is
/// valid. Call this with the complete array so null ancestors can hide placeholder values.
///
/// # Errors
///
/// Returns an error if a visible value is outside its extension type's domain or if executing
/// the array to read its values fails.
pub fn validate_extension_values(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    validate_visible(array, &Mask::new_true(array.len()), ctx)
}

fn contains_extension(dtype: &DType) -> bool {
    match dtype {
        DType::Extension(_) => true,
        DType::Struct(fields, _) => fields.field_dtypes().any(contains_extension),
        DType::List(element, _) | DType::FixedSizeList(element, ..) => contains_extension(element),
        DType::Map(map, _) => {
            contains_extension(&map.key_dtype()) || contains_extension(&map.value_dtype())
        }
        DType::Union(variants, _) => variants.variants().any(|dtype| contains_extension(&dtype)),
        _ => false,
    }
}

fn validate_visible(array: &ArrayRef, visible: &Mask, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    if visible.all_false() || !contains_extension(array.dtype()) {
        return Ok(());
    }
    if let Some(chunked) = array.as_opt::<Chunked>() {
        let mut start = 0;
        for chunk in chunked.iter_chunks() {
            let end = start + chunk.len();
            validate_visible(chunk, &visible.slice(start..end), ctx)?;
            start = end;
        }
        return Ok(());
    }

    let visible = array.validity()?.execute_mask(array.len(), ctx)? & visible;
    if visible.all_false() {
        return Ok(());
    }

    // Every row of a constant has the same value, so one row suffices.
    let (array, visible) = if array.is::<Constant>() {
        (array.slice(0..1)?, Mask::new_true(1))
    } else {
        (array.clone(), visible)
    };

    match array.execute::<Canonical>(ctx)? {
        Canonical::Extension(extension) => {
            let storage = extension.storage_array();
            validate_visible(storage, &visible, ctx)?;

            let visible_storage = if visible.all_true() {
                storage.clone()
            } else {
                storage.filter(visible)?
            };
            extension
                .ext_dtype()
                .validate_storage_array(&visible_storage, ctx)?;
        }
        Canonical::Struct(structure) => {
            for field in structure.iter_unmasked_fields() {
                validate_visible(field, &visible, ctx)?;
            }
        }
        Canonical::List(list) => {
            let elements = list.elements();
            if !elements.is_empty() {
                let visible = referenced_elements_mask(&list, &visible, ctx)?;
                validate_visible(elements, &visible, ctx)?;
            }
        }
        Canonical::FixedSizeList(list) => {
            let elements = list.elements();
            let visible =
                fixed_size_elements_visibility(&visible, list.list_size() as usize, elements.len());
            validate_visible(elements, &visible, ctx)?;
        }
        Canonical::Map(map) => validate_visible(map.entries(), &visible, ctx)?,
        Canonical::Union(union) => {
            let type_ids = union.type_ids().clone().execute::<PrimitiveArray>(ctx)?;
            let tags = Buffer::<u8>::from_byte_buffer(type_ids.buffer_handle().try_to_host_sync()?);
            for (&tag, child) in union
                .variants()
                .type_ids()
                .iter()
                .zip(union.iter_children())
            {
                if contains_extension(child.dtype()) {
                    let selected = Mask::from_buffer(
                        BitBufferMut::collect_bool(tags.len(), |row| tags[row] == tag).freeze(),
                    );
                    validate_visible(child, &(&visible & &selected), ctx)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn fixed_size_elements_visibility(visible: &Mask, size: usize, len: usize) -> Mask {
    match visible.slices() {
        AllOr::All => Mask::new_true(len),
        AllOr::None => Mask::new_false(len),
        AllOr::Some(ranges) => {
            let mut elements = BitBufferMut::new_unset(len);
            for &(start, end) in ranges {
                elements.fill_range(start * size, end * size, true);
            }
            Mask::from_buffer(elements.freeze())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::StructArray;
    use crate::dtype::Nullability;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::hidden(false)]
    #[case::visible(true)]
    fn list_views_validate_only_referenced_elements(#[case] visible: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = Time::try_new(TimeUnit::Seconds, Nullability::NonNullable)?.erased();
        let elements = buffer![
            86_401i32, // Unreferenced.
            7,         // Referenced by the second view.
            42,        // Shared by both views.
            86_401,    // Referenced by the first view.
        ]
        .into_array();
        let elements = ExtensionArray::try_new_structural(dtype, elements)?.into_array();
        let list = ListViewArray::try_new(
            elements,
            buffer![2u32, 1].into_array(),
            buffer![2u32, 2].into_array(),
            Validity::from_iter([visible, true]),
            &mut ctx,
        )?;

        let result = validate_extension_values(&list.into_array(), &mut ctx);
        if visible {
            let err = result.expect_err("a visible list element must be checked");
            assert!(err.to_string().contains("Invalid time scalar"), "{err}");
        } else {
            result?;
        }
        Ok(())
    }

    #[rstest]
    #[case::hidden(false)]
    #[case::visible(true)]
    fn fixed_size_lists_in_chunks_respect_ancestor_validity(
        #[case] visible: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = Time::try_new(TimeUnit::Seconds, Nullability::NonNullable)?.erased();
        let valid =
            ExtensionArray::try_new_structural(dtype.clone(), buffer![1i32, 2].into_array())?;
        let invalid =
            ExtensionArray::try_new_structural(dtype, buffer![3i32, 86_401].into_array())?;
        let lists = [valid, invalid].map(|elements| {
            FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 1).into_array()
        });
        let dtype = lists[0].dtype().clone();
        let lists = ChunkedArray::try_new(lists, dtype)?.into_array();
        let parent = StructArray::try_new(
            ["times"].into_iter().collect(),
            vec![lists],
            2,
            Validity::from_iter([true, visible]),
        )?;

        let result = validate_extension_values(&parent.into_array(), &mut ctx);
        if visible {
            let err = result.expect_err("a visible list element must be checked");
            assert!(err.to_string().contains("Invalid time scalar"), "{err}");
        } else {
            result?;
        }
        Ok(())
    }
}
