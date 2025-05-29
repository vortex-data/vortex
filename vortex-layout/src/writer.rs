use std::pin::Pin;

use vortex_error::VortexResult;

use crate::LayoutRef;

/// A future created by a strategy to yield a layout. It is its own
/// trait to be potentially extended with new methods.
// [layout writer]
pub trait LayoutWriter: Future<Output = VortexResult<LayoutRef>> {}
// [layout writer]

// Allow async blocks to impl LayoutWriter, this would change if more methods
// are added to LayoutWriter.
impl<F: Future<Output = VortexResult<LayoutRef>> + ?Sized + Send> LayoutWriter for F {}

pub type SendableLayoutWriter = Pin<Box<dyn LayoutWriter + Send>>;
