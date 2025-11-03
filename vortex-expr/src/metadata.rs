// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::display::{DisplayAs, DisplayFormat};
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use vortex_array::{DeserializeMetadata, SerializeMetadata};
use vortex_error::{vortex_bail, VortexResult};
use vortex_utils::dyn_traits::{DynEq, DynHash};

/// Trait for expression metadata.
pub trait ExprMetadata: 'static + Send + Sync + Debug + DynEq + DynHash + DisplayAs {
    fn as_any(&self) -> &dyn Any;
}

impl Hash for dyn ExprMetadata + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}

impl PartialEq for dyn ExprMetadata + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other.as_any())
    }
}
impl Eq for dyn ExprMetadata + '_ {}

/// Empty expression metadata.
///
/// Used when the expression is defined purely by its encoding ID.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EmptyMetadata;

impl ExprMetadata for EmptyMetadata {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl DisplayAs for EmptyMetadata {
    fn fmt_as(&self, _df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "")
    }
}
impl SerializeMetadata for EmptyMetadata {
    fn serialize(self) -> Vec<u8> {
        vec![]
    }
}
impl DeserializeMetadata for EmptyMetadata {
    type Output = EmptyMetadata;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        if !metadata.is_empty() {
            vortex_bail!("EmptyMetadata should not have metadata bytes")
        }
        Ok(EmptyMetadata)
    }
}
