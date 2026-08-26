// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core traits and types shared by Vortex layout implementations and consumers.
//!
//! Layout extension crates can depend on this crate for the typed [`Layout`] model, [`VTable`],
//! reader and writer traits, and segment IO abstractions without depending on the built-in layouts
//! and scan planner in `vortex-layout`.

pub mod display;
pub mod segments;
pub mod sequence;

pub use children::*;
pub use encoding::*;
pub use layout::*;
pub use reader::*;
pub use reader_context::*;
pub use strategy::*;
use vortex_session::registry::Interner;
pub use vtable::*;

mod children;
mod encoding;
mod flatbuffers;
mod layout;
mod reader;
mod reader_context;
mod strategy;
mod vtable;

/// Registry context used when serializing layouts.
pub type LayoutContext = Interner;
