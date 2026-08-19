// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Input decoding and output construction for row functions.
//!
//! [`ViewLen`] reports the rows addressable through a row-loop view. [`element`] defines the Rust
//! values decoded from input columns and built into simple output columns. [`sink`] handles outputs
//! that need row handles or batch-wide state. [`result`] defines immediate and deferred row
//! outcomes.

mod element;
pub use element::ElementTuple;
pub use element::IndexedElementTuple;
pub use element::InputElement;
pub use element::OutputElement;
pub(super) use element::batch_const;

mod result;
pub use result::FailureEvidence;
pub use result::SinkResult;

mod sink;
pub use sink::InitializedElement;
pub use sink::OutputSink;
pub use sink::UninitElementSink;

mod view;
pub use view::ViewLen;
