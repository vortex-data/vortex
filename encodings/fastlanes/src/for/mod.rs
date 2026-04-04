// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::FoRData;
pub use array::FoRArrayExt;

pub(crate) mod compute;

mod vtable;
pub use vtable::FoR;
pub use vtable::FoRArray;
