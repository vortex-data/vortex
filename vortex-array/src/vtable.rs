//! This module contains the VTable definitions for a Vortex encoding.

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{Array, Canonical};

/// The encoding [`VTable`] encapsulates _all_ logic for both an Array and an Encoding in a
/// single trait, giving users a single entry-point to implement their own arrays.
///
/// From this [`VTable`], we derive implementations for the [`Array`] and [`Encoding`] traits.
pub trait VTable: 'static {
    type Array: 'static + Send + Sync;
    type Encoding: 'static + Send + Sync; // We could use unstable and default this to ()

    // Encoding Functions

    /// Returns the ID of the encoding.
    fn id(encoding: &Self::Encoding) -> ArcRef<str>;

    /// Encodes a canonical array using this encoding.
    fn encode(
        encoding: &Self::Encoding,
        canonical: Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Self::Array>;

    // Array Functions

    fn len(array: &Self::Array) -> usize;

    fn dtype(array: &Self::Array) -> &DType;
}
