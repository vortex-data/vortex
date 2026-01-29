// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_dtype::ExtDType;
use vortex_dtype::Nullability;
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_error::VortexResult;

use crate::Scalar;
use crate::ScalarValue;

/// API for defining the scalar behavior of an extension DType.
pub trait ExtScalarVTable: ExtDTypeVTable {
    /// The native value type for this extension scalar.
    /// The `Default` trait should return a value representing `zero`.
    // TODO(ngates): require total ordering?
    type Value: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Hash;

    /// Return the `zero` value for this extension scalar.
    fn zero(&self, metadata: &Self::Metadata) -> Self::Value;

    /// Unpack the native value from the given scalar.
    ///
    /// Note that the storage scalar is guaranteed to be non-null.
    fn unpack(&self, dtype: &ExtDType<Self>, storage: &ScalarValue) -> VortexResult<Self::Value>;

    /// Pack the native value into the storage scalar.
    fn pack(
        &self,
        metadata: &Self::Metadata,
        value: Self::Value,
        nullability: Nullability,
    ) -> VortexResult<Scalar>;

    /// Pack a null value into the storage scalar.
    fn pack_null(&self, metadata: &Self::Metadata) -> VortexResult<Scalar>;
}
