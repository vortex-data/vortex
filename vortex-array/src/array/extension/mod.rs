use std::fmt::{Debug, Display};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use vortex_dtype::{DType, ExtDType, ExtID};
use vortex_error::{VortexExpect as _, VortexResult};

use crate::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use crate::encoding::ids;
use crate::stats::ArrayStatisticsCompute;
use crate::validity::{ArrayValidity, LogicalValidity};
use crate::variants::{ArrayVariants, ExtensionArrayTrait};
use crate::{impl_encoding, Array, ArrayDType, ArrayTrait, Canonical, IntoCanonical};

mod compute;

impl_encoding!("vortex.ext", ids::EXTENSION, Extension);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionMetadata;

impl Display for ExtensionMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ExtensionArray {
    pub fn new(ext_dtype: Arc<ExtDType>, storage: Array) -> Self {
        assert_eq!(
            ext_dtype.storage_dtype(),
            storage.dtype(),
            "ExtensionArray: storage_dtype must match storage array DType",
        );

        Self::try_from_parts(
            DType::Extension(ext_dtype),
            storage.len(),
            ExtensionMetadata,
            [storage].into(),
            Default::default(),
        )
        .vortex_expect("Invalid ExtensionArray")
    }

    pub fn storage(&self) -> Array {
        self.as_ref()
            .child(0, self.ext_dtype().storage_dtype(), self.len())
            .vortex_expect("Missing storage array for ExtensionArray")
    }

    #[allow(dead_code)]
    #[inline]
    pub fn id(&self) -> &ExtID {
        self.ext_dtype().id()
    }
}

impl ArrayTrait for ExtensionArray {}

impl ArrayVariants for ExtensionArray {
    fn as_extension_array(&self) -> Option<&dyn ExtensionArrayTrait> {
        Some(self)
    }
}

impl ExtensionArrayTrait for ExtensionArray {
    fn storage_array(&self) -> Array {
        self.storage()
    }
}

impl IntoCanonical for ExtensionArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        Ok(Canonical::Extension(self))
    }
}

impl ArrayValidity for ExtensionArray {
    fn is_valid(&self, index: usize) -> bool {
        self.storage().with_dyn(|a| a.is_valid(index))
    }

    fn logical_validity(&self) -> LogicalValidity {
        self.storage().with_dyn(|a| a.logical_validity())
    }
}

impl AcceptArrayVisitor for ExtensionArray {
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("storage", &self.storage())
    }
}

impl ArrayStatisticsCompute for ExtensionArray {
    // TODO(ngates): pass through stats to the underlying and cast the scalars.
}
