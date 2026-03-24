// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encapsulates a possibly lazy layout data structure, including the

use arcref::ArcRef;

mod children;
pub use children::*;

mod erased;
pub use erased::*;

mod plugin;
pub use plugin::*;

mod typed;
pub use typed::*;

mod flatbuffers;
pub mod session;
mod vtable;

pub use vtable::*;

/// A unique identifier for a layout.
pub type LayoutId = ArcRef<str>;

mod sealed {
    use crate::v2::layout::typed::Layout;
    use crate::v2::layout::vtable::LayoutVTable;

    pub(crate) trait Sealed {}

    impl<V: LayoutVTable> Sealed for Layout<V> {}
}
