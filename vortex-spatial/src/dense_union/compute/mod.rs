// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compute kernels for dense unions.
//!
//! Every kernel here is selector-only: it transforms the type IDs and offsets and retains each
//! compact child in full. That is O(selected rows) rather than O(child values), at the cost of
//! leaving unselected values behind and reordering per-child offsets.

mod filter;
mod mask;
mod slice;
mod take;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_error::VortexResult;

use super::DenseUnion;
use super::DenseUnionArrayExt;

/// Rebuild a dense union around transformed row selectors, retaining its compact children.
fn with_selectors(
    array: ArrayView<'_, DenseUnion>,
    type_ids: ArrayRef,
    offsets: ArrayRef,
) -> VortexResult<Option<ArrayRef>> {
    DenseUnion::try_new(
        type_ids,
        offsets,
        array.variants().clone(),
        array.iter_children().cloned(),
    )
    .map(|array| Some(array.into_array()))
}
