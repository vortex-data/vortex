// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod matcher;
mod vtable;

use std::any::Any;
use std::any::type_name;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::sync::Arc;

use arcref::ArcRef;
pub use matcher::*;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
pub use vtable::*;

use crate::DType;

/// A unique identifier for an extension type
pub type ExtID = ArcRef<str>;

/// An extension data type.
#[derive(Clone)]
pub struct ExtDType<V: VTable>(Arc<ExtDTypeAdapter<V>>);

impl<V: VTable> ExtDType<V> {
    /// Creates a new extension dtype with the given options and storage dtype.
    pub fn try_new(options: V::Options, storage_dtype: DType) -> VortexResult<Self> {
        V::validate(&options, &storage_dtype)?;
        Ok(Self(Arc::new(ExtDTypeAdapter::<V> {
            storage_dtype,
            options,
            vtable: PhantomData,
        })))
    }

    /// Returns the identifier of the extension type.
    pub fn id(&self) -> ExtID {
        self.0.id()
    }

    /// Returns the options of the extension type.
    pub fn options(&self) -> &V::Options {
        &self.0.options
    }

    /// Erase the concrete type information, returning a type-erased extension dtype.
    pub fn erase(self) -> ExtDTypeRef {
        ExtDTypeRef(self.0)
    }
}

/// Type-erased extension dtype - for heterogeneous storage
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtDTypeRef(Arc<dyn ExtDTypeImpl>);

impl ExtDTypeRef {
    /// Returns the identifier of the extension type.
    pub fn id(&self) -> ExtID {
        self.0.id()
    }

    /// Returns the untyped options of the extension type.
    pub fn options_ref(&self) -> ExtDTypeOptions<'_> {
        ExtDTypeOptions { ext_dtype: self }
    }

    /// Returns the storage dtype of the extension type.
    pub fn storage_dtype(&self) -> &DType {
        self.0.storage_dtype()
    }
}

/// Methods for downcasting type-erased extension dtypes.
impl ExtDTypeRef {
    /// Check if the extension dtype is of the concrete type.
    pub fn is<M: Matcher>(&self) -> bool {
        M::matches(&self)
    }

    /// Downcast to the concrete options type.
    pub fn try_options<M: Matcher>(&self) -> Option<M::Match<'_>> {
        M::try_match(&self)
    }

    /// Downcast to the concrete options type.
    pub fn options<M: Matcher>(&self) -> M::Match<'_> {
        self.try_options::<M>()
            .vortex_expect("Failed to downcast DynExtDType")
    }

    /// Downcast to the concrete options type.
    ///
    /// Returns `Err(self)` if the downcast fails.
    pub fn try_downcast<V: VTable>(self) -> Result<ExtDType<V>, ExtDTypeRef> {
        // Check if the concrete type matches
        if self.0.as_any().is::<ExtDTypeAdapter<V>>() {
            // SAFETY: type matches and ExtDTypeImpl<V> is the only implementor
            let ptr = Arc::into_raw(self.0) as *const ExtDTypeAdapter<V>;
            let inner = unsafe { Arc::from_raw(ptr) };
            Ok(ExtDType(inner))
        } else {
            Err(self)
        }
    }

    /// Downcast to the concrete options type.
    ///
    /// # Panics
    ///
    /// Panics if the downcast fails.
    pub fn downcast<V: VTable>(self) -> ExtDType<V> {
        self.try_downcast::<V>()
            .map_err(|this| {
                vortex_err!(
                    "Failed to downcast DynExtDType {} to {}",
                    this.0.id(),
                    type_name::<V>(),
                )
            })
            .vortex_expect("Failed to downcast DynExtDType")
    }
}

/// Wrapper for type-erased extension dtype options.
pub struct ExtDTypeOptions<'a> {
    pub(super) ext_dtype: &'a ExtDTypeRef,
}

impl ExtDTypeOptions<'_> {
    /// Serialize the options into a byte vector.
    pub fn serialize(&self) -> VortexResult<Vec<u8>> {
        self.ext_dtype.0.options_serialize()
    }
}

impl Display for ExtDTypeOptions<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ext_dtype.0.options_display(f)
    }
}

impl Debug for ExtDTypeOptions<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ext_dtype.0.options_debug(f)
    }
}

impl PartialEq for ExtDTypeOptions<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ext_dtype.0.options_eq(other.ext_dtype.0.options_any())
    }
}
impl Eq for ExtDTypeOptions<'_> {}

impl Hash for ExtDTypeOptions<'_> {
    fn hash<H: Hasher>(&self, mut state: &mut H) {
        self.ext_dtype.0.options_hash(&mut state);
    }
}

/// An object-safe trait encapsulating the behavior for extension DTypes.
trait ExtDTypeImpl: 'static + Send + Sync + private::Sealed {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> ExtID;
    fn storage_dtype(&self) -> &DType;
    fn options_any(&self) -> &dyn Any;
    fn options_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
    fn options_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
    fn options_eq(&self, other: &dyn Any) -> bool;
    fn options_hash(&self, state: &mut dyn Hasher);
    fn options_serialize(&self) -> VortexResult<Vec<u8>>;
}

struct ExtDTypeAdapter<V: VTable> {
    storage_dtype: DType,
    options: V::Options,
    vtable: PhantomData<V>,
}

impl<V: VTable> ExtDTypeImpl for ExtDTypeAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ExtID {
        V::id(&self.options)
    }

    fn storage_dtype(&self) -> &DType {
        &self.storage_dtype
    }

    fn options_any(&self) -> &dyn Any {
        &self.options
    }

    fn options_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        <V::Options as Debug>::fmt(&self.options, f)
    }

    fn options_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        <V::Options as Display>::fmt(&self.options, f)
    }

    fn options_eq(&self, other: &dyn Any) -> bool {
        let Some(other) = other.downcast_ref::<V::Options>() else {
            return false;
        };
        <V::Options as PartialEq>::eq(&self.options, other)
    }

    fn options_hash(&self, mut state: &mut dyn Hasher) {
        <V::Options as Hash>::hash(&self.options, &mut state);
    }

    fn options_serialize(&self) -> VortexResult<Vec<u8>> {
        V::serialize(&self.options)
    }
}

mod private {
    use super::ExtDTypeAdapter;

    pub trait Sealed {}
    impl<V: super::VTable> Sealed for ExtDTypeAdapter<V> {}
}
