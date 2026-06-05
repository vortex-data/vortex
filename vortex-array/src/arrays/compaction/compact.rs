// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ListViewArray;
use crate::arrays::StructArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::arrays::struct_::StructArrayExt;

/// Structurally compact a canonical array.
///
/// - `VarBinView` buffers are garbage collected via
///   [`compact_buffers`](crate::arrays::VarBinViewArray::compact_buffers).
/// - `List` (ListView) arrays are rebuilt to be zero-copy convertible to a `ListArray`
///   (overlaps removed, leading/trailing garbage trimmed), and their elements are recursively
///   compacted.
/// - `Struct` fields are recursively compacted.
/// - All other canonical arrays are returned unchanged.
///
/// Note that recursion bottoms out at scalar canonical arrays, so this terminates.
pub(crate) fn compact_canonical(
    canonical: Canonical,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    Ok(match canonical {
        Canonical::VarBinView(array) => array.compact_buffers()?.into_array(),
        Canonical::List(list_view) => compact_list_view(list_view, ctx)?,
        Canonical::Struct(struct_array) => compact_struct(struct_array, ctx)?,
        // TODO(joe): recurse into FixedSizeList elements and Extension storage.
        other => other.into_array(),
    })
}

fn compact_list_view(list_view: ListViewArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    // Make the list zero-copy convertible to a `ListArray` and trim unreferenced elements.
    let rebuilt = list_view.rebuild(ListViewRebuildMode::MakeExact)?;

    // Recursively compact the (now trimmed) element data. Compaction preserves logical length,
    // so the existing offsets and sizes remain valid.
    let elements = rebuilt.elements().clone().compact(ctx)?;
    if ArrayRef::ptr_eq(&elements, rebuilt.elements()) {
        return Ok(rebuilt.into_array());
    }

    // SAFETY: we only replace the elements child with a logically equivalent, equal-length
    // array, which preserves the zero-copy-to-list shape established by `MakeExact`.
    Ok(unsafe {
        ListViewArray::new_unchecked(
            elements,
            rebuilt.offsets().clone(),
            rebuilt.sizes().clone(),
            rebuilt.validity()?,
        )
        .with_zero_copy_to_list(rebuilt.is_zero_copy_to_list())
    }
    .into_array())
}

fn compact_struct(struct_array: StructArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let fields = struct_array.unmasked_fields();
    let mut new_fields = Vec::with_capacity(fields.len());
    let mut changed = false;
    for field in fields.iter() {
        let compacted = field.clone().compact(ctx)?;
        changed |= !ArrayRef::ptr_eq(&compacted, field);
        new_fields.push(compacted);
    }

    if !changed {
        return Ok(struct_array.into_array());
    }

    // SAFETY: each field is replaced with a logically equivalent, equal-length array, and the
    // struct's dtype and validity are preserved.
    Ok(unsafe {
        StructArray::new_unchecked(
            new_fields,
            struct_array.struct_fields().clone(),
            struct_array.len(),
            struct_array.struct_validity(),
        )
    }
    .into_array())
}
