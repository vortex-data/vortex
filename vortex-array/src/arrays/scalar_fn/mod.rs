// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub mod plugin;
mod rules;
mod vtable;

pub use array::ScalarFnArrayExt;
pub use vtable::ScalarFnFactoryExt;
pub use vtable::*;
