// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Return types for row closures.
//!
//! [`FailureEvidence`] represents deferred failures from owned row closures. [`SinkResult`] lets
//! the executor handle initialized sinks and sinks that require an [`InitializedElement`] or
//! [`InitializedRow`] token, with either infallible or immediate-error callbacks.

use std::ops::BitOrAssign;

use vortex_error::VortexResult;

use super::InitializedElement;
use super::InitializedRow;

/// Compact failure evidence that can be OR-reduced across rows.
///
/// [`Default::default`] **must** mean success, including for an empty batch. The compiler cannot
/// check this requirement.
pub trait FailureEvidence: Copy + Default + BitOrAssign {}

impl<T: Copy + Default + BitOrAssign> FailureEvidence for T {}

/// The result of writing one row: success or an immediate error.
///
/// This trait is sealed. Row functions choose one of its supplied implementations.
pub trait SinkResult: 'static + private::Sealed {
    /// The [`OutputSink::WriteToken`](super::OutputSink::WriteToken) carried by a success.
    type WriteToken: 'static;

    /// Whether this return type is infallible.
    const INFALLIBLE: bool;

    /// Convert this row's outcome into immediate success or failure.
    fn into_result(self) -> VortexResult<()>;
}

impl private::Sealed for () {}

impl SinkResult for () {
    type WriteToken = ();
    const INFALLIBLE: bool = true;

    fn into_result(self) -> VortexResult<()> {
        Ok(())
    }
}

impl private::Sealed for InitializedElement {}

impl SinkResult for InitializedElement {
    type WriteToken = InitializedElement;
    const INFALLIBLE: bool = true;

    fn into_result(self) -> VortexResult<()> {
        Ok(())
    }
}

impl private::Sealed for InitializedRow {}

impl SinkResult for InitializedRow {
    type WriteToken = InitializedRow;
    const INFALLIBLE: bool = true;

    fn into_result(self) -> VortexResult<()> {
        Ok(())
    }
}

impl private::Sealed for VortexResult<()> {}

impl SinkResult for VortexResult<()> {
    type WriteToken = ();
    const INFALLIBLE: bool = false;

    fn into_result(self) -> VortexResult<()> {
        self
    }
}

impl private::Sealed for VortexResult<InitializedElement> {}

impl SinkResult for VortexResult<InitializedElement> {
    type WriteToken = InitializedElement;
    const INFALLIBLE: bool = false;

    fn into_result(self) -> VortexResult<()> {
        self.map(|_| ())
    }
}

mod private {
    pub trait Sealed {}
}
