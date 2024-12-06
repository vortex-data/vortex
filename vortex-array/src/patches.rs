use serde::{Deserialize, Serialize};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, PType};
use vortex_error::VortexExpect;

use crate::{ArrayDType, ArrayData};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchesMetadata {
    len: usize,
    indices_ptype: PType,
}

impl PatchesMetadata {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn indices_dtype(&self) -> DType {
        DType::Primitive(self.indices_ptype, NonNullable)
    }
}

/// A helper for working with patched arrays.
pub struct Patches {
    indices: ArrayData,
    values: ArrayData,
}

impl Patches {
    pub fn new(indices: ArrayData, values: ArrayData) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "Patch indices and values must have the same length"
        );
        assert!(indices.dtype().is_int(), "Patch indices must be integers");
        Self { indices, values }
    }

    pub fn indices(&self) -> &ArrayData {
        &self.indices
    }

    pub fn values(&self) -> &ArrayData {
        &self.values
    }

    pub fn metadata(&self) -> PatchesMetadata {
        PatchesMetadata {
            len: self.indices.len(),
            indices_ptype: PType::try_from(self.indices.dtype()).vortex_expect("primitive indices"),
        }
    }
}
