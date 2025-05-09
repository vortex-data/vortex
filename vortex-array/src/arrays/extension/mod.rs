use std::sync::Arc;

use vortex_dtype::{DType, ExtDType, ExtID};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::array::{ArrayCanonicalImpl, ArrayValidityImpl};
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::VTableRef;
use crate::{
    Array, ArrayImpl, ArrayOperationsImpl, ArrayRef, ArrayStatisticsImpl, Canonical, EmptyMetadata,
    Encoding,
};

mod compute;
mod serde;

#[derive(Clone, Debug)]
pub struct ExtensionArray {
    dtype: DType,
    storage: ArrayRef,
    stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct ExtensionEncoding;
impl Encoding for ExtensionEncoding {
    type Array = ExtensionArray;
    type Metadata = EmptyMetadata;
}

impl ExtensionArray {
    pub fn new(ext_dtype: Arc<ExtDType>, storage: ArrayRef) -> Self {
        assert_eq!(
            ext_dtype.storage_dtype(),
            storage.dtype(),
            "ExtensionArray: storage_dtype must match storage array DType",
        );
        Self {
            dtype: DType::Extension(ext_dtype),
            storage,
            stats_set: ArrayStats::default(),
        }
    }

    pub fn ext_dtype(&self) -> &Arc<ExtDType> {
        let DType::Extension(ext) = &self.dtype else {
            unreachable!("ExtensionArray: dtype must be an ExtDType")
        };
        ext
    }

    pub fn storage(&self) -> &ArrayRef {
        &self.storage
    }

    #[allow(dead_code)]
    #[inline]
    pub fn id(&self) -> &ExtID {
        self.ext_dtype().id()
    }
}

impl ArrayImpl for ExtensionArray {
    type Encoding = ExtensionEncoding;

    fn _len(&self) -> usize {
        self.storage.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&ExtensionEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        Ok(Self::new(self.ext_dtype().clone(), children[0].clone()))
    }
}

impl ArrayStatisticsImpl for ExtensionArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayCanonicalImpl for ExtensionArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        Ok(Canonical::Extension(self.clone()))
    }
}

impl ArrayOperationsImpl for ExtensionArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(
            ExtensionArray::new(self.ext_dtype().clone(), self.storage().slice(start, stop)?)
                .into_array(),
        )
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::extension(
            self.ext_dtype().clone(),
            self.storage().scalar_at(index)?,
        ))
    }
}

impl ArrayValidityImpl for ExtensionArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.storage.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.storage.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.storage.all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.storage.validity_mask()
    }
}
