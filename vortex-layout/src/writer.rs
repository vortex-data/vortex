use std::pin::Pin;

use vortex_error::VortexResult;

use crate::LayoutRef;

pub trait LayoutWriter: Future<Output = VortexResult<LayoutRef>> {}

// Allow async blocks to impl LayoutWriter, this would change if more methods
// are added to LayoutWriter.
impl<F: Future<Output = VortexResult<LayoutRef>> + ?Sized + Send> LayoutWriter for F {}

pub type SendableLayoutWriter = Pin<Box<dyn LayoutWriter + Send>>;
