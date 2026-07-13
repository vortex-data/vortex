// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::ListViewArrayExt;
pub use array::ListViewData;
pub use array::ListViewDataParts;
pub use vtable::ListViewArray;

pub(crate) mod compute;

mod vtable;
pub use vtable::ListView;

pub(crate) fn initialize(session: &VortexSession) {
    vtable::initialize(session);
}

mod conversion;
pub use conversion::list_from_list_view;
pub use conversion::list_view_from_list;
pub use conversion::recursive_list_from_list_view;

mod rebuild;
pub use rebuild::DEFAULT_REBUILD_DENSITY_THRESHOLD;
pub use rebuild::DEFAULT_TRIM_ELEMENTS_THRESHOLD;
pub use rebuild::ListViewRebuildMode;
use vortex_session::VortexSession;

#[cfg(test)]
mod tests;
