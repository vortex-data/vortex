// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar functions computed one row at a time.
//!
//! Start with [`RowFn`] to define an operation and [`RowVisitor`] to select one of its output
//! capabilities. [`InputElement`] and [`ElementTuple`] describe how columns become row values.
//! [`OutputElement`] covers independent owned values, while [`OutputSink`] supports outputs that
//! need row handles or shared batch state. [`SinkResult`] describes how a sink-writing closure
//! reports errors.
//!
//! The internal executor owns decoding, batch constants, null propagation, allocation, and
//! validity. A visitor's prepare closure may derive shared state from constant operands once per
//! batch.

mod execute;
pub use execute::RowExecution;

mod batch;

mod row_fn;
pub use row_fn::RowFn;

mod types;
pub use types::ElementTuple;
pub use types::IndexedElementTuple;
pub use types::InitializedElement;
pub use types::InputElement;
pub use types::OutputElement;
pub use types::OutputSink;
pub use types::SinkResult;
pub use types::UninitElementSink;

mod visitor;
pub use visitor::RowVisitor;

mod vtable;
