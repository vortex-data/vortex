// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::RLEArray;
pub use array::RLEArrayExt;

mod compute;
mod kernel;

mod vtable;
pub use vtable::RLEVTable;
