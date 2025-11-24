// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::DeltaArray;
pub use array::delta_compress::delta_compress;

mod compute;

mod vtable;
pub use vtable::DeltaVTable;
