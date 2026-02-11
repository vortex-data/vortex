// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition of [`ExtScalarVTable`].

use std::fmt::Debug;
use std::fmt::Display;

use vortex_dtype::DType;
use vortex_dtype::ExtDTypeRef;
use vortex_dtype::ExtID;
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ScalarValue;
use crate::extension::ExtScalarValue;
use crate::extension::ExtScalarValueRef;

/// API for defining the scalar behavior of an extension DType.
pub trait ExtScalarVTable: ExtDTypeVTable {
    /// Extract a native Rust value type from a storage value.
    ///
    /// The value only represents non-null values. We denote nullable values as `Option<Value>`.
    type Value<'a>: Display;

    /// Unpack a native value from the storage ScalarValue.
    ///
    /// This call is infallible assuming the [`ExtScalarVTable::validate_scalar_value`] function has
    /// been called previously.
    fn unpack<'a>(
        &self,
        metadata: &'a <Self as ExtDTypeVTable>::Metadata,
        storage_dtype: &'a DType,
        storage_value: &'a ScalarValue,
    ) -> Self::Value<'a>;

    /// Validate the given storage value is compatible with the extension type.
    ///
    /// Note that [`ExtDTypeVTable::validate_dtype`] is always called first to validate the storage
    /// [`DType`].
    ///
    /// # Errors
    ///
    /// Returns an error if the storage [`ScalarValue`] is not compatible with the extension type.
    fn validate_scalar_value(
        &self,
        metadata: &<Self as ExtDTypeVTable>::Metadata,
        storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> VortexResult<()>;
}

/// A dynamic vtable for extension scalars, used for type-erased deserialization.
pub trait DynExtScalarVTable: 'static + Send + Sync + Debug {
    /// Returns the ID for this extension type.
    fn id(&self) -> ExtID;

    /// Creates a [`ExtScalarValueRef`] from a [`ScalarValue`] and a type-erased [`ExtDTypeRef`].
    ///
    /// The reason we need this is that the internal implementation is the only thing that can
    /// access the [`ExtScalarVTable`].
    ///
    /// # Errors
    ///
    /// Returns an error if the storage value is not compatible with the extension type.
    fn build(
        &self,
        ext_dtype: ExtDTypeRef,
        storage: ScalarValue,
    ) -> VortexResult<ExtScalarValueRef>;
}

impl<V: ExtScalarVTable> DynExtScalarVTable for V {
    fn id(&self) -> ExtID {
        ExtDTypeVTable::id(self)
    }

    fn build(
        &self,
        ext_dtype: ExtDTypeRef,
        storage: ScalarValue,
    ) -> VortexResult<ExtScalarValueRef> {
        let typed_ext_dtype = ext_dtype
            .try_downcast::<V>()
            .map_err(|_| vortex_err!("unable to downcast the extension dtype with the vtable"))?;

        let owned = ExtScalarValue::<V>::try_new(typed_ext_dtype, storage)?;
        Ok(owned.erased())
    }
}
