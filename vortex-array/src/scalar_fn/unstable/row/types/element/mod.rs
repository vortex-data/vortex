// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The element types a row function can read and produce.
//!
//! [`InputElement::Elem`] can borrow from its decoded column. Owned row computations return an
//! [`OutputElement`]. Runtime-shaped outputs use an
//! [`OutputSink`](crate::scalar_fn::unstable::row::OutputSink).

mod bool;

mod input;
pub use input::InputElement;

mod output;
pub use output::OutputElement;

mod primitive;

mod tuple;
pub use tuple::ElementTuple;
pub use tuple::IndexedElementTuple;
pub use tuple::batch_const;
pub(in crate::scalar_fn::unstable::row) use tuple::decoded_source;
