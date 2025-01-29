//! This module contains the VTable definitions for a Vortex Array.

use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

mod canonical;
mod compute;
mod metadata;
mod statistics;
mod validate;
mod validity;
mod variants;
mod visitor;

pub use canonical::*;
pub use compute::*;
pub use metadata::*;
pub use statistics::*;
pub use validate::*;
pub use validity::*;
pub use variants::*;
pub use visitor::*;

use crate::encoding::EncodingId;
use crate::ArrayData;

/// Dyn-compatible VTable trait for a Vortex array encoding.
///
/// This trait provides extension points for arrays to implement various features of Vortex.
/// It is split into multiple sub-traits to make it easier for consumers to break up the
/// implementation, as well as to allow for optional implementation of certain features, for example
/// compute functions.
///
/// It is recommended that you use [`crate::impl_encoding`] to assist in writing a new
/// array encoding.
pub trait EncodingVTable:
    'static
    + Sync
    + Send
    + Debug
    + CanonicalVTable
    + ComputeVTable
    + MetadataVTable<ArrayData>
    + StatisticsVTable<ArrayData>
    + ValidateVTable<ArrayData>
    + ValidityVTable<ArrayData>
    + VariantsVTable<ArrayData>
    + VisitorVTable<ArrayData>
{
    /// Return the ID for this encoding implementation.
    fn id(&self) -> EncodingId;

    /// Return a reference to this encoding as a `dyn Any` for type erasure.
    fn as_any(&self) -> &dyn Any;
}

impl PartialEq for dyn EncodingVTable + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn EncodingVTable + '_ {}

impl Hash for dyn EncodingVTable + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state)
    }
}
