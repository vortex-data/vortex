// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod patch;

pub use array::*;

pub mod compute;

mod vtable;
pub use vtable::{BoolEncoding, BoolMaskedValidityRule, BoolVTable};

#[cfg(feature = "test-harness")]
mod test_harness;
