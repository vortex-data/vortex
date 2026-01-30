// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::FilterArray;
pub use array::FilterArrayParts;

mod execute;

mod rules;

mod vtable;
pub use vtable::FilterVTable;
