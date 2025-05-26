use vortex_error::VortexResult;

use crate::LayoutRef;

pub trait LayoutWriter: Future<Output = VortexResult<LayoutRef>> + Send {}

// Allow async blocks to impl LayoutWriter, this would change if more methods
// are added to LayoutWriter.
impl<F: Future<Output = VortexResult<LayoutRef>> + ?Sized + Send> LayoutWriter for F {}
