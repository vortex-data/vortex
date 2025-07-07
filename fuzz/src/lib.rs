// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![feature(error_generic_member_access)]

mod array;
pub mod error;
mod file;

pub use array::{Action, ExpectedValue, FuzzArrayAction, sort_canonical_array};
pub use file::FuzzFileAction;
