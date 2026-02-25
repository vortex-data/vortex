// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::vortex_err;

use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::ScalarValue;
use crate::scalar::extension::ExtScalarValue;
use crate::scalar::extension::typed::DynExtScalarValue;
use crate::scalar::extension::typed::ExtScalarValueInner;

/// A type-erased extension scalar value.
///
/// This is the extension scalar analog of [`ExtDTypeRef`]: it stores an [`ExtVTable`]
/// and a storage [`ScalarValue`] behind a trait object, allowing heterogeneous storage inside
/// `ScalarValue::Extension` (so that we do not need a generic parameter).
///
/// You can use [`try_downcast()`] or [`downcast()`] to recover the concrete vtable type as an
/// [`ExtScalarValue<V>`].
///
/// [`ExtDTypeRef`]: crate::dtype::extension::ExtDTypeRef
/// [`try_downcast()`]: ExtScalarValueRef::try_downcast
/// [`downcast()`]: ExtScalarValueRef::downcast
#[derive(Clone)]
pub struct ExtScalarValueRef(pub(super) Arc<dyn DynExtScalarValue>);

// NB: If you need access to the vtable, you probably want to add a method and implementation to
// `ExtScalarValueInnerImpl` and `ExtScalarValueInner`.
/// Methods for downcasting type-erased extension scalars.
impl ExtScalarValueRef {
    /// Returns the [`ExtId`] identifying this extension scalar's type.
    pub fn id(&self) -> ExtId {
        self.0.id()
    }

    /// Returns a reference to the underlying storage [`ScalarValue`].
    pub fn storage_value(&self) -> &ScalarValue {
        self.0.storage_value()
    }

    /// Attempts to downcast to a concrete [`ExtScalarValue<V>`].
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if the underlying vtable type does not match `V`.
    pub fn try_downcast<V: ExtVTable>(self) -> Result<ExtScalarValue<V>, ExtScalarValueRef> {
        // `ExtScalarValueInner<V>` is the only implementor of `ExtScalarValueInnerImpl` (due to
        // the sealed implementation below), so if the vtable is correct, we know the type can be
        // downcasted and reinterpreted safely.
        if !self.0.as_any().is::<ExtScalarValueInner<V>>() {
            return Err(self);
        }

        let ptr = Arc::into_raw(self.0) as *const ExtScalarValueInner<V>;
        // SAFETY: We verified the type matches above, so the size and alignment are correct.
        let inner = unsafe { Arc::from_raw(ptr) };

        Ok(ExtScalarValue(inner))
    }

    /// Downcasts to a concrete [`ExtScalarValue<V>`].
    ///
    /// # Panics
    ///
    /// Panics if the underlying vtable type does not match `V`.
    pub fn downcast<V: ExtVTable>(self) -> ExtScalarValue<V> {
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
}

impl fmt::Display for ExtScalarValueRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.0.id(), self.0.storage_value())
    }
}

impl fmt::Debug for ExtScalarValueRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtScalar")
            .field("id", &self.0.id())
            .field("storage_value", self.0.storage_value())
            .finish()
    }
}

// TODO(connor): In the future we may want to allow implementors to customize this behavior.

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
