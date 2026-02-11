// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Type-erased and typed extension scalar values.
//!
//! Extension scalars wrap an underlying storage [`ScalarValue`] together with an
//! [`ExtScalarVTable`] that gives the value semantic meaning beyond its raw storage
//! representation.
//!
//! There are two main (public) types:
//!
//! - [`ExtScalarValueRef`]: A type-erased extension scalar that can be stored heterogeneously.
//! - [`OwnedExtScalarValue`]: A typed extension scalar parameterized by a concrete
//!   [`ExtScalarVTable`] implementation.
//!
//! These mirror the [`ExtDTypeRef`] / [`ExtDType`] pattern in `vortex-dtype`.
//!
//! [`ExtDType`]: vortex_dtype::ExtDType

use std::any::Any;
use std::any::type_name;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_dtype::ExtDType;
use vortex_dtype::ExtDTypeRef;
use vortex_dtype::ExtID;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ScalarValue;
use crate::extension::ExtScalarVTable;

/// A type-erased extension scalar value.
///
/// This is the extension scalar analog of [`ExtDTypeRef`]: it stores an [`ExtScalarVTable`]
/// and a storage [`ScalarValue`] behind a trait object, allowing heterogeneous storage inside
/// [`ScalarValue::Extension`] (so that we do not need a generic parameter).
///
/// You can use [`try_downcast`] or [`downcast`] to recover the concrete vtable type as an
/// [`OwnedExtScalarValue<V>`].
///
/// [`try_downcast`]: ExtScalarValueRef::try_downcast
/// [`downcast`]: ExtScalarValueRef::downcast
#[derive(Clone)]
pub struct ExtScalarValueRef(Arc<dyn DynExtScalarValue>);

/// A typed extension scalar value, parameterized by a concrete [`ExtScalarVTable`].
///
/// This is the extension scalar analog of [`ExtDType<V>`]: it retains full type information
/// about the vtable, providing direct access to the vtable and storage value without
/// downcasting.
///
/// You can construct one of these via [`try_new`] from an [`ExtDType<V>`] and a storage
/// [`ScalarValue`], and erase the type with [`erased`] to obtain an [`ExtScalarValueRef`].
///
/// [`ExtDType<V>`]: vortex_dtype::ExtDType
/// [`try_new`]: OwnedExtScalarValue::try_new
/// [`erased`]: OwnedExtScalarValue::erased
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtScalarValue<V: ExtScalarVTable>(Arc<ExtScalarValueAdapter<V>>);

/// The concrete inner representation of an extension scalar, pairing a vtable with its storage
/// value.
///
/// This is the sole implementor of [`ExtScalarValueAdapterImpl`], enabling [`ExtScalarValueRef`] to
/// downcast back to the concrete vtable type via [`Any`].
#[derive(Debug, PartialEq, Eq, Hash)]
struct ExtScalarValueAdapter<V: ExtScalarVTable> {
    /// The extension scalar vtable.
    vtable: V,
    /// The underlying storage value.
    storage: ScalarValue,
}

impl<V: ExtScalarVTable> DynExtScalarValue for ExtScalarValueAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ExtID {
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

    fn fmt_ext_scalar(&self, ext_dtype: &ExtDTypeRef, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = V::unpack(
            &self.vtable,
            ext_dtype.metadata::<V>(),
            ext_dtype.storage_dtype(),
            &self.storage,
        );

        write!(f, "{value}")
    }
}

// NB: If you need access to the vtable, you probably want to add a method and implementation to
// `ExtScalarValueAdapterImpl` and `ExtScalarValueAdapter`.
/// Methods for downcasting type-erased extension scalars.
impl ExtScalarValueRef {
    /// Returns the [`ExtID`] identifying this extension scalar's type.
    pub fn id(&self) -> ExtID {
        self.0.id()
    }

    /// Returns a reference to the underlying storage [`ScalarValue`].
    pub fn storage_value(&self) -> &ScalarValue {
        self.0.storage_value()
    }

    /// Formats the extension scalar using the provided [`ExtDTypeRef`] for metadata context.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying [`fmt::Write`] operation fails.
    pub fn fmt_ext_scalar(
        &self,
        ext_dtype: &ExtDTypeRef,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        self.0.fmt_ext_scalar(ext_dtype, f)
    }

    /// Attempts to downcast to a concrete [`OwnedExtScalarValue<V>`].
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if the underlying vtable type does not match `V`.
    pub fn try_downcast<V: ExtScalarVTable>(self) -> Result<ExtScalarValue<V>, ExtScalarValueRef> {
        // `ExtScalarValueAdapter<V>` is the only implementor of `ExtScalarValueAdapterImpl` (due to
        // the sealed implementation below), so if the vtable is correct, we know the type can be
        // downcasted and reinterpreted safely.
        if !self.0.as_any().is::<ExtScalarValueAdapter<V>>() {
            return Err(self);
        }

        let ptr = Arc::into_raw(self.0) as *const ExtScalarValueAdapter<V>;
        // SAFETY: We verified the type matches above, so the size and alignment are correct.
        let inner = unsafe { Arc::from_raw(ptr) };

        Ok(ExtScalarValue(inner))
    }

    /// Downcasts to a concrete [`OwnedExtScalarValue<V>`].
    ///
    /// # Panics
    ///
    /// Panics if the underlying vtable type does not match `V`.
    pub fn downcast<V: ExtScalarVTable>(self) -> ExtScalarValue<V> {
        self.try_downcast::<V>()
            .map_err(|this| {
                vortex_err!(
                    "Failed to downcast ExtScalar {} to {}",
                    this.0.id(),
                    type_name::<V>(),
                )
            })
            .vortex_expect("Failed to downcast ExtScalar")
    }

    /// Attempts to downcast the vtable to a concrete [`ExtScalarVTable`] type by reference.
    ///
    /// Unlike [`try_downcast`], this borrows rather than consuming `self`.
    ///
    /// [`try_downcast`]: ExtScalarValueRef::try_downcast
    pub fn try_get_vtable<V: ExtScalarVTable>(&self) -> Option<&V> {
        self.0.vtable_any().downcast_ref::<V>()
    }

    /// Downcasts the vtable to a concrete [`ExtScalarVTable`] type by reference.
    ///
    /// Unlike [`downcast`], this borrows rather than consuming `self`.
    ///
    /// # Panics
    ///
    /// Panics if the underlying vtable type does not match `V`.
    ///
    /// [`downcast`]: ExtScalarValueRef::downcast
    pub fn get_vtable<V: ExtScalarVTable>(&self) -> &V {
        self.try_get_vtable::<V>()
            .vortex_expect("ExtScalarVTable downcast failed")
    }

    /// Checks whether this extension scalar value is compatible with the given [`ExtDTypeRef`].
    ///
    /// This validates that the vtable types match and that the storage value passes the
    /// vtable's [`ExtScalarVTable::validate_scalar_value`] check.
    ///
    /// # Errors
    ///
    /// Returns an error if it is not compatible with the extension type.
    pub fn validate(&self, ext_dtype: &ExtDTypeRef) -> VortexResult<()> {
        self.0.validate(ext_dtype)
    }
}

impl<V: ExtScalarVTable> ExtScalarValue<V> {
    /// Creates a new extension scalar from a storage [`ScalarValue`], validating it against the
    /// given [`ExtDType`].
    ///
    /// # Errors
    ///
    /// Returns an error if [`ExtScalarVTable::validate_scalar_value`] fails for the given
    /// storage value and extension dtype.
    pub fn try_new(ext_dtype: ExtDType<V>, storage: ScalarValue) -> VortexResult<Self> {
        ExtScalarVTable::validate_scalar_value(
            ext_dtype.vtable(),
            ext_dtype.metadata(),
            ext_dtype.storage_dtype(),
            &storage,
        )?;

        Ok(Self(Arc::new(ExtScalarValueAdapter::<V> {
            vtable: ext_dtype.vtable().clone(),
            storage,
        })))
    }

    /// Erases the concrete type information, returning a type-erased [`ExtScalarValueRef`].
    pub fn erased(self) -> ExtScalarValueRef {
        ExtScalarValueRef(self.0)
    }

    /// Returns the [`ExtID`] identifying this extension scalar's type.
    pub fn id(&self) -> ExtID {
        self.0.vtable.id()
    }

    /// Returns a reference to the [`ExtScalarVTable`] for this extension scalar.
    pub fn vtable(&self) -> &V {
        &self.0.vtable
    }

    /// Returns a reference to the underlying storage [`ScalarValue`].
    pub fn storage_value(&self) -> &ScalarValue {
        self.0.storage_value()
    }

    // pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
    //     if self.value.is_none() && !dtype.is_nullable() {
    //         vortex_bail!(
    //             "cannot cast extension dtype with id {} and storage type {} to {}",
    //             self.ext_dtype.id(),
    //             self.ext_dtype.storage_dtype(),
    //             dtype
    //         );
    //     }
    //
    //     if self.ext_dtype.storage_dtype().eq_ignore_nullability(dtype) {
    //         // Casting from an extension type to the underlying storage type is OK.
    //         return Ok(Scalar::new(dtype.clone(), self.value.clone()));
    //     }
    //
    //     if let DType::Extension(ext_dtype) = dtype
    //         && self.ext_dtype.eq_ignore_nullability(ext_dtype)
    //     {
    //         return Ok(Scalar::new(dtype.clone(), self.value.clone()));
    //     }
    //
    //     vortex_bail!(
    //         "cannot cast extension dtype with id {} and storage type {} to {}",
    //         self.ext_dtype.id(),
    //         self.ext_dtype.storage_dtype(),
    //         dtype
    //     );
    // }
}

impl Display for ExtScalarValueRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.0.id(), self.0.storage_value())
    }
}

impl Debug for ExtScalarValueRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtScalar")
            .field("id", &self.0.id())
            .field("storage_value", self.0.storage_value())
            .finish()
    }
}

impl PartialEq for ExtScalarValueRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id() && self.0.storage_value() == other.0.storage_value()
    }
}
impl Eq for ExtScalarValueRef {}

impl PartialOrd for ExtScalarValueRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.storage_value().partial_cmp(other.0.storage_value())
    }
}

impl Hash for ExtScalarValueRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state);
        self.0.storage_value().hash(state);
    }
}

/// Private module to seal [`ExtScalarValueAdapterImpl`].
///
/// Note that this is not strictly necessary since [`ExtScalarValueAdapter`] and
/// [`ExtScalarValueAdapterImpl`] are both private to this module, this is just for hygiene.
mod sealed {
    use super::*;

    /// Marker trait to prevent external implementations of [`ExtScalarValueAdapterImpl`].
    pub(super) trait Sealed {}

    /// This can be the **only** implementor for [`ExtScalarValueAdapterImpl`].
    impl<V: ExtScalarVTable> Sealed for ExtScalarValueAdapter<V> {}
}

/// An object-safe, sealed trait encapsulating the behavior for extension scalar values.
///
/// This mirrors [`ExtDTypeImpl`] in `vortex-dtype`: it provides type-erased access to the
/// extension scalar's identity, storage value, and display formatting. The only implementor is
/// [`ExtScalarValueAdapter`].
///
/// [`ExtDTypeImpl`]: vortex_dtype::extension
trait DynExtScalarValue: sealed::Sealed + 'static + Send + Sync {
    /// Returns `self` as a trait object for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Returns the [`ExtID`] identifying this extension type.
    fn id(&self) -> ExtID;
    /// Returns the vtable as a trait object for downcasting.
    fn vtable_any(&self) -> &dyn Any;
    /// Returns a reference to the underlying storage [`ScalarValue`].
    fn storage_value(&self) -> &ScalarValue;
    /// Checks whether this extension scalar value is compatible with the given [`ExtDTypeRef`].
    fn validate(&self, ext_dtype: &ExtDTypeRef) -> VortexResult<()>;
    /// Formats the extension scalar using the provided [`ExtDTypeRef`] for metadata context.
    fn fmt_ext_scalar(&self, ext_dtype: &ExtDTypeRef, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}
