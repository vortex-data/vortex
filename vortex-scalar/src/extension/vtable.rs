// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_dtype::ExtDType;
use vortex_dtype::ExtDTypeRef;
use vortex_dtype::ExtID;
use vortex_dtype::Nullability;
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExtScalar;
use crate::ExtScalarRef;
use crate::Scalar;
use crate::ScalarValue;

/// API for defining the scalar behavior of an extension DType.
pub trait ExtScalarVTable: ExtDTypeVTable {
    /// The native value type for this extension scalar.
    /// The `Default` trait should return a value representing `zero`.
    // TODO(ngates): require total ordering?
    type Value: 'static + Send + Sync + Clone + Debug + Display + Eq + Hash;

    /// Unpack the native value from the given scalar.
    ///
    /// Note that the storage scalar is guaranteed to be non-null.
    fn unpack(&self, dtype: &ExtDType<Self>, storage: &ScalarValue) -> VortexResult<Self::Value>;

    /// Pack the native value into the storage scalar.
    /// FIXME(ngates): do we take ExtDType here and use its storage DType?
    fn pack(
        &self,
        metadata: &Self::Metadata,
        value: Option<&Self::Value>,
        nullability: Nullability,
    ) -> VortexResult<Scalar>;
}

/// A dynamic vtable for extension scalars, used for type-erased deserialization.
pub trait DynExtScalarVTable: 'static + Send + Sync + Debug {
    /// Returns the ID for this extension type.
    fn id(&self) -> ExtID;

    /// Unpack an extension scalar from a scalar value.
    fn unpack(&self, dtype: &ExtDTypeRef, storage: &ScalarValue) -> VortexResult<ExtScalarRef>;
}

impl<V: ExtScalarVTable> DynExtScalarVTable for V {
    fn id(&self) -> ExtID {
        ExtDTypeVTable::id(self)
    }

    fn unpack(&self, dtype: &ExtDTypeRef, storage: &ScalarValue) -> VortexResult<ExtScalarRef> {
        let dtype = dtype
            .clone()
            .try_downcast::<V>()
            .map_err(|_| vortex_err!("DTypeRef is not of expected extension type {}", self.id()))?;

        let value = match storage {
            ScalarValue::Null => None,
            _ => Some(ExtScalarVTable::unpack(self, &dtype, storage)?),
        };

        Ok(ExtScalar::try_with_vtable(self.clone(), dtype.clone(), value)?.erased())
    }
}
