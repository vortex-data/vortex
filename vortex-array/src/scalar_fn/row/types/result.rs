// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What a sink-writing row closure may return.

use vortex_error::VortexResult;

use super::InitializedElement;

mod private {
    pub trait Sealed {}
}

/// The result of writing one row: success or an immediate error.
///
/// This trait is sealed; row functions choose one of its supplied implementations.
pub trait SinkResult: 'static + private::Sealed {
    /// The [`OutputSink::WriteToken`](super::OutputSink::WriteToken) carried by a success.
    type WriteToken: 'static;

    /// Loop-local state used while accumulating row results.
    type Accumulated: 'static + Copy + Default;

    /// Whether this return type can carry an error.
    const FALLIBLE: bool;

    /// Merge this row's outcome into the batch-wide reduction.
    fn accumulate(self, accumulated: &mut Self::Accumulated) -> VortexResult<()>;
}

impl private::Sealed for () {}

impl SinkResult for () {
    type WriteToken = ();
    type Accumulated = ();

    const FALLIBLE: bool = false;

    fn accumulate(self, _accumulated: &mut ()) -> VortexResult<()> {
        Ok(())
    }
}

impl private::Sealed for InitializedElement {}

impl SinkResult for InitializedElement {
    type WriteToken = InitializedElement;
    type Accumulated = ();

    const FALLIBLE: bool = false;

    fn accumulate(self, _accumulated: &mut ()) -> VortexResult<()> {
        Ok(())
    }
}

impl private::Sealed for VortexResult<()> {}

impl SinkResult for VortexResult<()> {
    type WriteToken = ();
    type Accumulated = ();

    const FALLIBLE: bool = true;

    fn accumulate(self, _accumulated: &mut ()) -> VortexResult<()> {
        self
    }
}

impl private::Sealed for VortexResult<InitializedElement> {}

impl SinkResult for VortexResult<InitializedElement> {
    type WriteToken = InitializedElement;
    type Accumulated = ();

    const FALLIBLE: bool = true;

    fn accumulate(self, _accumulated: &mut ()) -> VortexResult<()> {
        self.map(|_| ())
    }
}
