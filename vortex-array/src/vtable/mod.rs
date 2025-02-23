//! This module contains the VTable definitions for a Vortex Array.

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

mod compute;
mod serde;
mod statistics;

pub use compute::*;
pub use serde::*;
pub use statistics::*;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::encoding::EncodingId;
use crate::serde::ArrayParts;
use crate::{Array, ArrayRef, Encoding};

/// A reference to an array VTable, either static or arc'd.
#[derive(Debug, Clone)]
pub struct VTableRef(Inner);

#[derive(Debug, Clone)]
enum Inner {
    Static(&'static dyn EncodingVTable),
    Arc(Arc<dyn EncodingVTable>),
}

impl VTableRef {
    pub const fn from_static(vtable: &'static dyn EncodingVTable) -> Self {
        VTableRef(Inner::Static(vtable))
    }

    pub fn from_arc(vtable: Arc<dyn EncodingVTable>) -> Self {
        VTableRef(Inner::Arc(vtable))
    }
}

impl Deref for VTableRef {
    type Target = dyn EncodingVTable;

    fn deref(&self) -> &Self::Target {
        match &self.0 {
            Inner::Static(vtable) => *vtable,
            Inner::Arc(vtable) => vtable.deref(),
        }
    }
}

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
    + ComputeVTable
    + for<'a> SerdeVTable<&'a dyn Array>
    + for<'a> StatisticsVTable<&'a dyn Array>
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

impl Debug for dyn EncodingVTable + '_ {
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}
