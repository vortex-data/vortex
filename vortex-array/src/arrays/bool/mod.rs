// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod patch;

pub use array::BoolArrayParts;
pub use array::BoolData;

pub(crate) mod compute;

mod vtable;
pub use compute::rules::BoolMaskedValidityRule;
pub use vtable::Bool;
pub use vtable::BoolArray;

#[cfg(feature = "_test-harness")]
mod test_harness;
