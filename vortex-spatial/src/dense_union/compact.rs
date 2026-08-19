// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::UnionVariants;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::tag_lookup;

/// A dense union compacted into a directly Arrow-exportable layout.
pub(crate) struct CompactUnion {
    variants: UnionVariants,
    /// The row-aligned Arrow type IDs.
    pub type_ids: Vec<i8>,
    /// The row-aligned offsets into [`Self::children`], increasing within each child.
    pub offsets: Vec<i32>,
    /// The compacted children in variant order, each holding exactly the rows that select it.
    children: Vec<ArrayRef>,
    /// The row validity, which Arrow instead expresses through the selected child.
    pub validity: Mask,
}

impl CompactUnion {
    /// Return the compacted child selected by a data-level type tag.
    pub(crate) fn child(&self, tag: u8) -> Option<&ArrayRef> {
        self.variants
            .tag_to_child_index(tag)
            .and_then(|child_index| self.children.get(child_index))
    }
}

/// Compact a dense union so that its offsets increase within each child.
///
/// Vortex's selector-only operations reorder and repeat per-child offsets and retain unselected
/// child rows, so the compact children are not an Arrow dense-union layout as they stand. This
/// gathers each child down to exactly the rows that select it, in row order, and rebases the
/// offsets onto the result.
///
/// A row's nullity moves from the union onto the selected child, matching Arrow's dense union,
/// which has no validity of its own.
///
/// # Errors
///
/// Returns an error for unknown type IDs on valid rows, for offsets that exceed `i32`, or when
/// gathering a child fails.
pub(crate) fn compact_for_arrow(
    variants: UnionVariants,
    type_ids: &PrimitiveArray,
    offsets: &PrimitiveArray,
    children: Vec<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<CompactUnion> {
    let validity = type_ids.validity()?.execute_mask(type_ids.len(), ctx)?;
    let all_valid = validity.all_true();
    let type_id_values = type_ids.as_slice::<u8>();
    let offset_values = offsets.as_slice::<i32>();

    let child_indices = tag_lookup(&variants);
    // A null row is free to carry a type ID no variant declares: canonicalization and `scalar_at`
    // both skip a null row's selectors without reading them. Arrow has no such slack, so park
    // those rows on the first variant, where the null index below makes the value null.
    let fallback_child = variants
        .type_ids()
        .first()
        .and_then(|tag| child_indices[usize::from(*tag)])
        .ok_or_else(|| vortex_err!("DenseUnion has no variants"))?;

    let mut selections = vec![Vec::<Option<i32>>::new(); children.len()];
    let mut arrow_type_ids = Vec::with_capacity(type_id_values.len());
    let mut arrow_offsets = Vec::with_capacity(offset_values.len());
    for (row, ((type_id, offset), valid)) in type_id_values
        .iter()
        .zip(offset_values)
        .zip(validity.iter())
        .enumerate()
    {
        let child_index = match (child_indices[usize::from(*type_id)], valid) {
            (Some(child_index), _) => child_index,
            (None, false) => fallback_child,
            (None, true) => vortex_bail!("DenseUnion row has unknown type ID {type_id}"),
        };
        let arrow_type_id = variants.child_index_to_tag(child_index);
        arrow_type_ids.push(
            i8::try_from(arrow_type_id)
                .map_err(|_| vortex_err!("DenseUnion type ID {arrow_type_id} exceeds i8"))?,
        );
        arrow_offsets.push(
            i32::try_from(selections[child_index].len())
                .map_err(|_| vortex_err!("DenseUnion child offset exceeds i32 at row {row}"))?,
        );
        selections[child_index].push(valid.then_some(*offset));
    }

    let children = children
        .into_iter()
        .zip(selections)
        .map(|(child, selection)| {
            // A null index gathers to a null value, which is how the row's nullity reaches the
            // child. Keeping the child non-nullable when nothing selected it while null lets the
            // Arrow export match a non-nullable child field.
            let indices = if all_valid || selection.iter().all(Option::is_some) {
                PrimitiveArray::from_iter(selection.into_iter().flatten()).into_array()
            } else {
                PrimitiveArray::from_option_iter(selection).into_array()
            };
            child.take(indices)
        })
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(CompactUnion {
        variants,
        type_ids: arrow_type_ids,
        offsets: arrow_offsets,
        children,
        validity,
    })
}
