// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;

use vortex_dtype::DType;
use vortex_dtype::ExtID;
use vortex_dtype::extension::ExtDTypeVTable;

use crate::ScalarValue;

/// API for defining the scalar behavior of an extension DType.
pub trait ExtScalarVTable: ExtDTypeVTable {
    /// Format the Scalar value for [`fmt::Display`].
    fn fmt_scalar(
        &self,
        metadata: &Self::Metadata,
        storage_dtype: &DType,
        storage_value: &ScalarValue,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result;

    /// Validate the given storage value is compatible with the extension type.
    /// Note that [`ExtDTypeVTable::validate_dtype`] is called first to validate the storage DType.
    fn validate_scalar(
        &self,
        metadata: &Self::Metadata,
        storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> vortex_error::VortexResult<()>;
}

/// A dynamic vtable for extension scalars, used for type-erased deserialization.
pub trait DynExtScalarVTable: 'static + Send + Sync + Debug {
    /// Returns the ID for this extension type.
    fn id(&self) -> ExtID;

    // Unpack an extension scalar from a scalar value.
    // fn unpack(&self, dtype: &ExtDTypeRef, storage: &ScalarValue) -> VortexResult<ExtScalarRef>;
}

impl<V: ExtScalarVTable> DynExtScalarVTable for V {
    fn id(&self) -> ExtID {
        ExtDTypeVTable::id(self)
    }
    //
    // fn unpack(&self, dtype: &ExtDTypeRef, storage: &ScalarValue) -> VortexResult<ExtScalarRef> {
    //     let dtype = dtype
    //         .clone()
    //         .try_downcast::<V>()
    //         .map_err(|_| vortex_err!("DTypeRef is not of expected extension type {}", self.id()))?;
    //
    //     let value = match storage {
    //         ScalarValue::Null => None,
    //         _ => Some(ExtScalarVTable::unpack(self, &dtype, storage)?),
    //     };
    //
    //     Ok(ExtScalar::try_with_vtable(self.clone(), dtype.clone(), value)?.erased())
    // }
}
