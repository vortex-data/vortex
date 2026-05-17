// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod mask;
pub(crate) mod rules;
mod slice;
pub(crate) mod take;

/// The threshold below which we rebuild the elements of a listview.
///
/// We don't touch `elements` on the metadata-only path since reorganizing it can be expensive.
/// However, we also don't want to drag around a large amount of garbage data when the selection
/// is sparse. Below this fraction of list rows retained, the rebuild is worth it.
/// Rebuilding is needed when exporting the ListView's elements.
///
// TODO(connor)[ListView]: Ideally, we would only rebuild after all `take`s and `filter`
//  compute functions have run, at the "top" of the operator tree. However, we cannot do this
//  right now, so we will just rebuild every time (similar to [`ListArray`]).
pub(crate) const REBUILD_DENSITY_THRESHOLD: f32 = 0.1;
