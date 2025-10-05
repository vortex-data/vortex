// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::{BoolArray, BooleanBufferExt};
pub use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};

pub mod compute;

mod vtable;
pub use vtable::{BoolEncoding, BoolVTable};

#[cfg(feature = "test-harness")]
mod test_harness;
