// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::ExtensionData;
pub use vtable::ExtensionArray;

pub(crate) mod compute;

mod vtable;
pub use vtable::Extension;
