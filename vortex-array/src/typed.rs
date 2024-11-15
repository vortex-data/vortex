use std::sync::{Arc, OnceLock};

use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_panic, VortexError, VortexResult};

use crate::stats::StatsSet;
use crate::{
    ArrayData, ArrayDef, InnerArrayData, IntoArrayData, OwnedArrayData, ToArrayData,
    TryDeserializeArrayMetadata,
};

/// Container for an array with all the associated implementation type information (encoding reference and ID, actual array type, metadata type).
#[derive(Debug, Clone)]
pub struct TypedArray<D: ArrayDef> {
    array: ArrayData,
    lazy_metadata: OnceLock<D::Metadata>,
}

impl<D: ArrayDef> TypedArray<D> {
    pub fn try_from_parts(
        dtype: DType,
        len: usize,
        metadata: D::Metadata,
        buffer: Option<Buffer>,
        children: Arc<[ArrayData]>,
        stats: StatsSet,
    ) -> VortexResult<Self> {
        let array = OwnedArrayData::try_new(
            D::ENCODING,
            dtype,
            len,
            Arc::new(metadata),
            buffer,
            children,
            stats,
        )?
        .into();
        Ok(Self {
            array,
            lazy_metadata: OnceLock::new(),
        })
    }

    pub fn metadata(&self) -> &D::Metadata {
        match &self.array.0 {
            InnerArrayData::Owned(d) => d
                .metadata()
                .as_any()
                .downcast_ref::<D::Metadata>()
                .unwrap_or_else(|| {
                    vortex_panic!(
                        "Failed to downcast metadata to {} for typed array with ID {} and encoding {}",
                        std::any::type_name::<D::Metadata>(),
                        D::ID.as_ref(),
                        D::ENCODING.id().as_ref(),
                    )
                }),
            InnerArrayData::Viewed(v) => self
                .lazy_metadata
                .get_or_init(|| {
                    D::Metadata::try_deserialize_metadata(v.metadata()).unwrap_or_else(|err| {
                        vortex_panic!(
                            "Failed to deserialize ArrayView metadata for typed array with ID {} and encoding {}: {}",
                            D::ID.as_ref(),
                            D::ENCODING.id().as_ref(),
                            err
                        )
                    })
                }),
        }
    }
}

impl<D: ArrayDef> TypedArray<D> {
    pub fn array(&self) -> &ArrayData {
        &self.array
    }
}

impl<D: ArrayDef> TryFrom<ArrayData> for TypedArray<D> {
    type Error = VortexError;

    fn try_from(array: ArrayData) -> Result<Self, Self::Error> {
        if array.encoding().id() != D::ENCODING.id() {
            vortex_bail!(
                "incorrect encoding {}, expected {}",
                array.encoding().id().as_ref(),
                D::ENCODING.id().as_ref(),
            );
        }
        Ok(Self {
            array,
            lazy_metadata: OnceLock::new(),
        })
    }
}

impl<'a, D: ArrayDef> TryFrom<&'a ArrayData> for TypedArray<D> {
    type Error = VortexError;

    fn try_from(value: &'a ArrayData) -> Result<Self, Self::Error> {
        value.clone().try_into()
    }
}

impl<D: ArrayDef> ToArrayData for TypedArray<D> {
    fn to_array(&self) -> ArrayData {
        self.array.clone()
    }
}

impl<D: ArrayDef> IntoArrayData for TypedArray<D> {
    fn into_array(self) -> ArrayData {
        self.array
    }
}
