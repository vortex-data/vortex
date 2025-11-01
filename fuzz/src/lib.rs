// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// VortexFuzzError is quite large, but we don't care about the performance impact for fuzzing.
#![allow(clippy::result_large_err)]

mod array;
pub mod error;
mod file;

pub use array::{sort_canonical_array, Action, CompressorStrategy, ExpectedValue, FuzzArrayAction};
pub use file::FuzzFileAction;
use std::sync::LazyLock;
use vortex::VortexSessionDefault;
use vortex_session::VortexSession;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);
