// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The element types a row function can read and produce.
//!
//! [`InputElement::Elem`] can borrow from its decoded column. Owned row computations return an
//! [`OutputElement`]. Runtime-shaped outputs use an
//! [`OutputSink`](crate::scalar_fn::unstable::row::OutputSink).
//!
//! Booleans and primitives implement both traits directly. UTF-8 columns decode through
//! [`Utf8Column`] and build through [`Utf8Sink`](crate::scalar_fn::unstable::row::Utf8Sink).

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

mod utf8;
pub use utf8::Utf8Column;
pub use utf8::Utf8View;
