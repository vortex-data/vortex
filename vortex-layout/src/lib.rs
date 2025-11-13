// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod layouts;

pub use children::*;
pub use encoding::*;
pub use flatbuffers::*;
#[cfg(gpu_unstable)]
pub use gpu::*;
pub use layout::*;
pub use reader::*;
pub use strategy::*;
use vortex_array::VTableContext;
pub use vtable::*;
pub mod aliases;
mod children;
pub mod display;
mod encoding;
mod flatbuffers;
#[cfg(gpu_unstable)]
pub mod gpu;
mod layout;
mod reader;
pub mod segments;
pub mod sequence;
pub mod session;
mod strategy;
pub mod vtable;

pub type LayoutContext = VTableContext<LayoutEncodingRef>;
