use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_error::VortexResult;

use crate::{LayoutData, LayoutReader};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayoutId(pub u16);

impl Display for LayoutId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

pub trait LayoutEncoding: Debug + Send + Sync {
    /// Returns the globally unique ID for this type of layout.
    fn id(&self) -> LayoutId;

    /// Construct a [`LayoutReader`] for the provided [`LayoutData`].
    ///
    /// May panic if the provided `LayoutData` is not the same encoding as this `LayoutEncoding`.
    fn reader(&self, layout: LayoutData, ctx: ContextRef) -> VortexResult<Arc<dyn LayoutReader>>;
}

pub type LayoutEncodingRef = &'static dyn LayoutEncoding;
