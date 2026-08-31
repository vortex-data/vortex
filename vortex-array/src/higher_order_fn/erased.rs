// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::debug_with::DebugWith;

use crate::higher_order_fn::HigherOrderFnId;
use crate::higher_order_fn::HigherOrderFnOptions;
use crate::higher_order_fn::HigherOrderFnVTable;
use crate::higher_order_fn::TypedHigherOrderFnInstance;
use crate::higher_order_fn::typed::DynHigherOrderFn;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;

/// A type-erased higher-order function, pairing a vtable with per-call options.
#[derive(Clone)]
pub struct HigherOrderFnRef(pub(super) Arc<dyn DynHigherOrderFn>);

impl HigherOrderFnRef {
    /// Bind `vtable` with its per-call `options`.
    pub fn new<V: HigherOrderFnVTable>(vtable: V, options: V::Options) -> Self {
        TypedHigherOrderFnInstance::new(vtable, options).erased()
    }

    /// The function's global identifier.
    pub fn id(&self) -> HigherOrderFnId {
        self.0.id()
    }

    /// Whether this function uses vtable `V`.
    pub fn is<V: HigherOrderFnVTable>(&self) -> bool {
        self.0.as_any().is::<TypedHigherOrderFnInstance<V>>()
    }

    /// Return typed options when this function uses vtable `V`.
    pub fn as_opt<V: HigherOrderFnVTable>(&self) -> Option<&V::Options> {
        self.0
            .as_any()
            .downcast_ref::<TypedHigherOrderFnInstance<V>>()
            .map(TypedHigherOrderFnInstance::options)
    }

    /// Return typed options for vtable `V`.
    ///
    /// # Panics
    ///
    /// Panics if this function does not use vtable `V`.
    pub fn as_<V: HigherOrderFnVTable>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("higher-order function options type mismatch")
    }

    /// Return these options behind an opaque type-erased handle.
    pub fn options(&self) -> HigherOrderFnOptions<'_> {
        HigherOrderFnOptions { inner: &*self.0 }
    }

    /// Return the arity of the ordinary arguments.
    pub fn arity(&self) -> Arity {
        self.0.arity()
    }

    /// Return the number of lambda arguments.
    pub fn lambda_arity(&self) -> usize {
        self.0.lambda_arity()
    }

    /// Return the name of an ordinary argument.
    pub fn child_name(&self, child_idx: usize) -> ChildName {
        self.0.child_name(child_idx)
    }

    /// Serialize the per-call options.
    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.0.options_serialize()
    }

    /// Downcast this function to its typed representation.
    pub fn try_downcast<V: HigherOrderFnVTable>(
        self,
    ) -> Result<Arc<TypedHigherOrderFnInstance<V>>, Self> {
        if self.is::<V>() {
            let ptr = Arc::into_raw(self.0) as *const TypedHigherOrderFnInstance<V>;
            Ok(unsafe { Arc::from_raw(ptr) })
        } else {
            Err(self)
        }
    }

    /// Downcast this function to its typed representation.
    ///
    /// # Panics
    ///
    /// Panics if this function does not use vtable `V`.
    pub fn downcast<V: HigherOrderFnVTable>(self) -> Arc<TypedHigherOrderFnInstance<V>> {
        self.try_downcast::<V>()
            .map_err(|function| {
                vortex_err!(
                    "failed to downcast higher-order function {} to {}",
                    function.id(),
                    type_name::<V>(),
                )
            })
            .vortex_expect("failed to downcast higher-order function")
    }
}

impl Debug for HigherOrderFnRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HigherOrderFnRef")
            .field("vtable", &self.id())
            .field("options", &DebugWith(|fmt| self.0.options_debug(fmt)))
            .finish()
    }
}

impl Display for HigherOrderFnRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.id())?;
        self.0.options_display(f)?;
        write!(f, ")")
    }
}

impl PartialEq for HigherOrderFnRef {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id() && self.0.options_eq(other.0.options_any())
    }
}

impl Eq for HigherOrderFnRef {}

impl Hash for HigherOrderFnRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
        self.0.options_hash(state);
    }
}
