// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::scalar_fn::typed::DynScalarFn;

/// Scalar function options with their type erased.
///
/// An [`Expression`](crate::expr::Expression) node holds its function's options this way, next to
/// the function's id, until binding looks the function up in a session and checks that the
/// options are its [`ScalarFnVTable::Options`](crate::scalar_fn::ScalarFnVTable::Options) type.
///
/// Equality and hashing dispatch on the concrete type, so options of different types are never
/// equal.
#[derive(Clone, Debug)]
pub struct ErasedOptions(Arc<dyn DynOptions>);

impl ErasedOptions {
    /// Erases `options`. Erasing options that are already erased returns them as they are.
    pub fn new<O>(options: O) -> Self
    where
        O: 'static + Send + Sync + Debug + Display + PartialEq + Eq + Hash,
    {
        if let Some(erased) = (&options as &dyn Any).downcast_ref::<Self>() {
            return erased.clone();
        }
        Self(Arc::new(options))
    }

    /// The options as [`Any`], for downcasting to their concrete type.
    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }

    /// The options if they are of type `O`.
    pub fn downcast_ref<O: 'static>(&self) -> Option<&O> {
        self.as_any().downcast_ref::<O>()
    }
}

impl Display for ErasedOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl PartialEq for ErasedOptions {
    fn eq(&self, other: &Self) -> bool {
        self.0.dyn_eq(other.as_any())
    }
}

impl Eq for ErasedOptions {}

impl Hash for ErasedOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.dyn_hash(state);
    }
}

/// Object-safe view of an options value, with equality and hashing dispatched by type.
trait DynOptions: Send + Sync + Debug + Display {
    fn as_any(&self) -> &dyn Any;
    fn dyn_eq(&self, other: &dyn Any) -> bool;
    fn dyn_hash(&self, hasher: &mut dyn Hasher);
}

impl<O> DynOptions for O
where
    O: 'static + Send + Sync + Debug + Display + PartialEq + Eq + Hash,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dyn_eq(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<O>().is_some_and(|o| self == o)
    }

    fn dyn_hash(&self, mut hasher: &mut dyn Hasher) {
        self.hash(&mut hasher);
    }
}

/// An opaque handle to expression options.
pub struct ScalarFnOptions<'a> {
    pub(super) inner: &'a dyn DynScalarFn,
}

impl Display for ScalarFnOptions<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.inner.options_display(f)
    }
}

impl Debug for ScalarFnOptions<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.inner.options_debug(f)
    }
}

impl PartialEq for ScalarFnOptions<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.id() == other.inner.id() && self.inner.options_eq(other.inner.options_any())
    }
}
impl Eq for ScalarFnOptions<'_> {}

impl Hash for ScalarFnOptions<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.id().hash(state);
        self.inner.options_hash(state);
    }
}

impl ScalarFnOptions<'_> {
    /// Serialize the options to a byte vector.
    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.inner.options_serialize()
    }

    /// Clone the options into their type-erased form.
    pub fn to_erased(&self) -> ErasedOptions {
        self.inner.options_erased()
    }
}

impl<'a> ScalarFnOptions<'a> {
    /// Return the underlying `Any` reference.
    pub fn as_any(&self) -> &'a dyn Any {
        self.inner.options_any()
    }
}
