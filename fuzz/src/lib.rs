// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::use_debug)]

mod array;
pub mod error;
mod file;

use std::sync::LazyLock;

pub use array::{Action, CompressorStrategy, ExpectedValue, FuzzArrayAction, sort_canonical_array};
pub use file::FuzzFileAction;
use vortex::VortexSessionDefault;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

pub static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);
pub static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::default().with_handle(RUNTIME.handle()));
