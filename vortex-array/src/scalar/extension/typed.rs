// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::ScalarValue;
use crate::scalar::extension::ExtScalarValueRef;

/// A typed extension scalar value, parameterized by a concrete [`ExtVTable`].
///
/// This is the extension scalar analog of [`ExtDType<V>`]: it retains full type information
/// about the vtable, providing direct access to the vtable and storage value without
/// downcasting.
///
/// You can construct one of these via [`try_new()`] from an [`ExtDType<V>`] and a storage
/// [`ScalarValue`], and erase the type with [`erased()`] to obtain an [`ExtScalarValueRef`].
///
/// [`try_new()`]: ExtScalarValue::try_new
/// [`erased()`]: ExtScalarValue::erased
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtScalarValue<V: ExtVTable>(pub(super) Arc<ExtScalarValueInner<V>>);

/// The concrete inner representation of an extension scalar, pairing a vtable with its storage
/// value.
///
/// This is the sole implementor of [`DynExtScalarValue`], enabling [`ExtScalarValueRef`] to
/// downcast back to the concrete vtable type via [`Any`].
#[derive(Debug, PartialEq, Eq, Hash)]
pub(super) struct ExtScalarValueInner<V: ExtVTable> {
    /// The extension scalar vtable.
    vtable: V,
    /// The underlying storage value.
    storage: ScalarValue,
}

impl<V: ExtVTable> DynExtScalarValue for ExtScalarValueInner<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ExtId {
        self.vtable.id()
    }

    fn vtable_any(&self) -> &dyn Any {
        &self.vtable
    }

    fn storage_value(&self) -> &ScalarValue {
        &self.storage
    }

    fn validate(&self, ext_dtype: &ExtDTypeRef) -> VortexResult<()> {
        // Downcasting the metadata implicitly verifies the vtable types match, but we still want to
        // validate the actual scalar value.
        let Some(metadata) = ext_dtype.metadata_opt::<V>() else {
            vortex_bail!("extension scalar is not compatible with type {ext_dtype}");
        };

        self.vtable
            .validate_scalar_value(metadata, ext_dtype.storage_dtype(), &self.storage)
    }

    fn fmt_ext_scalar_value(
        &self,
        ext_dtype: &ExtDTypeRef,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let value = V::unpack_native(
            &self.vtable,
            ext_dtype.metadata::<V>(),
            ext_dtype.storage_dtype(),
            &self.storage,
        )
        .vortex_expect("invalid extension dtype for this extension scalar value");

        write!(f, "{value}")
    }
}

/// An object-safe, sealed trait encapsulating the behavior for extension scalar values.
///
/// This mirrors [`DynExtDType`] in `vortex-dtype`: it provides type-erased access to the
/// extension scalar's identity, storage value, and display formatting. The only implementor is
/// [`ExtScalarValueInner`].
///
/// [`DynExtDType`]: crate::dtype::extension::DynExtDType
pub(super) trait DynExtScalarValue: super::sealed::Sealed + 'static + Send + Sync {
    /// Returns `self` as a trait object for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Returns the [`ExtID`] identifying this extension type.
    fn id(&self) -> ExtId;
    /// Returns the vtable as a trait object for downcasting.
    fn vtable_any(&self) -> &dyn Any;
    /// Returns a reference to the underlying storage [`ScalarValue`].
    fn storage_value(&self) -> &ScalarValue;
    /// Checks whether this extension scalar value is compatible with the given [`ExtDTypeRef`].
    fn validate(&self, ext_dtype: &ExtDTypeRef) -> VortexResult<()>;
    /// Formats the extension scalar using the provided [`ExtDTypeRef`] for metadata context.
    fn fmt_ext_scalar_value(
        &self,
        ext_dtype: &ExtDTypeRef,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result;
}

impl<V: ExtVTable> ExtScalarValue<V> {
    /// Creates a new extension scalar from a storage [`ScalarValue`], validating it against the
    /// given [`ExtDType`].
    ///
    /// # Errors
    ///
    /// Returns an error if [`ExtVTable::validate_scalar_value`] fails for the given
    /// storage value and extension dtype.
    pub fn try_new(ext_dtype: ExtDType<V>, storage: ScalarValue) -> VortexResult<Self> {
        ext_dtype.vtable().validate_scalar_value(
            ext_dtype.metadata(),
            ext_dtype.storage_dtype(),
            &storage,
        )?;

        Ok(Self(Arc::new(ExtScalarValueInner::<V> {
            vtable: ext_dtype.vtable().clone(),
            storage,
        })))
    }

    /// Erases the concrete type information, returning a type-erased [`ExtScalarValueRef`].
    pub fn erased(self) -> ExtScalarValueRef {
        ExtScalarValueRef(self.0)
    }

    /// Returns the [`ExtId`] identifying this extension scalar's type.
    pub fn id(&self) -> ExtId {
        self.0.vtable.id()
    }

    /// Returns a reference to the [`ExtVTable`] for this extension scalar.
    pub fn vtable(&self) -> &V {
        &self.0.vtable
    }

    /// Returns a reference to the underlying storage [`ScalarValue`].
    pub fn storage_value(&self) -> &ScalarValue {
        self.0.storage_value()
    }
}
