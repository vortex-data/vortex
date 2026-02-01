// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_dtype::ExtDType;
use vortex_dtype::ExtID;
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_error::VortexResult;

use crate::ScalarValue;

/// API for defining the scalar behavior of an extension DType.
pub trait ExtScalarVTable: ExtDTypeVTable {
    /// The native value type for this extension scalar.
    /// The `Default` trait should return a value representing `zero`.
    // TODO(ngates): require total ordering?
    type Value: 'static + Send + Sync + Clone + Debug + Display + Eq + PartialOrd + Hash;

    /// Unpack the native value from the given scalar.
    ///
    /// Note that the storage scalar value is guaranteed to be non-null.
    fn unpack(&self, dtype: &ExtDType<Self>, storage: &ScalarValue) -> VortexResult<Self::Value>;

    /// Pack the native value into the storage scalar.
    fn pack(&self, dtype: &ExtDType<Self>, value: &Self::Value) -> VortexResult<ScalarValue>;

    /// Validate that the given storage value is compatible with the extension type.
    fn validate(&self, value: &Self::Value, ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        Self::pack(self, ext_dtype, value).map(|_| ())
    }
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
