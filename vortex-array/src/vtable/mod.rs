//! This module contains the VTable definitions for a Vortex Array.

use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

mod compute;
mod serde;
mod statistics;

pub use compute::*;
pub use serde::*;
pub use statistics::*;

use crate::arcref::ArcRef;
use crate::encoding::EncodingId;
use crate::{Array, Encoding};

/// A reference to an array VTable, either static or arc'd.
pub type VTableRef = ArcRef<dyn EncodingVTable>;

/// Dyn-compatible VTable trait for a Vortex array encoding.
///
/// This trait provides extension points for arrays to implement various features of Vortex.
/// It is split into multiple sub-traits to make it easier for consumers to break up the
/// implementation, as well as to allow for optional implementation of certain features, for example
/// compute functions.
pub trait EncodingVTable:
    'static
    + Sync
    + Send
    + ComputeVTable
    + for<'a> SerdeVTable<&'a dyn Array>
    + for<'a> StatisticsVTable<&'a dyn Array>
{
    /// Return the ID for this encoding implementation.
    fn id(&self) -> EncodingId;
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

impl Debug for dyn EncodingVTable + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}

impl Display for dyn EncodingVTable + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}

impl<
    E: Encoding
        + ComputeVTable
        + for<'a> SerdeVTable<&'a dyn Array>
        + for<'a> StatisticsVTable<&'a dyn Array>,
> EncodingVTable for E
{
    fn id(&self) -> EncodingId {
        E::ID
    }
}
