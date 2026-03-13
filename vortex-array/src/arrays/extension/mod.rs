// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::ExtensionArray;

mod view;
pub use view::ExtArray;

pub(crate) mod compute;

mod vtable;

pub use vtable::Extension;
