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
    /// Extract a native Rust value type from a storage value.
    ///
    /// The value must be able to represent nulls if the storage value is `None`.
    /// If this isn't required, a good default would be to use [`Option<&ScalarValue>`].
    type Value<'a>: 'static + Send + Sync;

    /// Unpack a native value from the storage ScalarValue.
    ///
    /// This call is infallible assuming the [`ExtScalarVTable::validate_scalar`] function has
    /// been called previously.
    fn unpack(
        &self,
        metadata: &Self::Metadata,
        storage_dtype: &DType,
        storage_value: Option<&ScalarValue>,
    ) -> Self::Value<'_>;

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
}

impl<V: ExtScalarVTable> DynExtScalarVTable for V {
    fn id(&self) -> ExtID {
        ExtDTypeVTable::id(self)
    }
}
