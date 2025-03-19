//! This module contains the VTable definitions for a Vortex Array.

use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

mod compute;

pub use compute::*;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arcref::ArcRef;
use crate::encoding::EncodingId;
use crate::serde::ArrayParts;
use crate::{ArrayContext, ArrayRef};

/// A reference to an array VTable, either static or arc'd.
pub type VTableRef = ArcRef<dyn EncodingVTable>;

/// Dyn-compatible VTable trait for a Vortex array encoding.
///
/// This trait provides extension points for arrays to implement various features of Vortex.
/// It is split into multiple sub-traits to make it easier for consumers to break up the
/// implementation, as well as to allow for optional implementation of certain features, for example
/// compute functions.
pub trait EncodingVTable: 'static + Sync + Send + ComputeVTable {
    /// Return the ID for this encoding implementation.
    fn id(&self) -> EncodingId;

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        _dtype: DType,
        _len: usize,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "Decoding not supported for encoding {}",
            ctx.lookup_encoding(parts.encoding_id())
                .vortex_expect("Encoding already validated")
                .id()
        )
    }
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
