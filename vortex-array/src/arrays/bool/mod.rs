// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod patch;

pub use array::BoolArray;
pub use array::BoolArrayParts;

pub(crate) mod compute;

mod vtable;
pub use compute::rules::BoolMaskedValidityRule;
pub use vtable::BoolVTable;

#[cfg(feature = "_test-harness")]
mod test_harness;
