// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::higher_order_fn::HigherOrderFnId;
use crate::higher_order_fn::HigherOrderFnRef;
use crate::higher_order_fn::HigherOrderFnVTable;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;

/// A typed higher-order-function instance, pairing a vtable with per-call options.
pub struct TypedHigherOrderFnInstance<V: HigherOrderFnVTable> {
    vtable: V,
    options: V::Options,
}

impl<V: HigherOrderFnVTable> TypedHigherOrderFnInstance<V> {
    /// Create a typed higher-order-function instance.
    pub fn new(vtable: V, options: V::Options) -> Self {
        Self { vtable, options }
    }

    /// Return the vtable.
    pub fn vtable(&self) -> &V {
        &self.vtable
    }

    /// Return the typed options.
    pub fn options(&self) -> &V::Options {
        &self.options
    }

    /// Erase the concrete type information.
    pub fn erased(self) -> HigherOrderFnRef {
        HigherOrderFnRef(Arc::new(self))
    }
}

pub(super) trait DynHigherOrderFn: 'static + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> HigherOrderFnId;
    fn options_any(&self) -> &dyn Any;
    fn arity(&self) -> Arity;
    fn lambda_arity(&self) -> usize;
    fn child_name(&self, child_idx: usize) -> ChildName;
    fn options_serialize(&self) -> VortexResult<Option<Vec<u8>>>;
    fn options_eq(&self, other_options: &dyn Any) -> bool;
    fn options_hash(&self, hasher: &mut dyn Hasher);
    fn options_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
    fn options_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
}

impl<V: HigherOrderFnVTable> DynHigherOrderFn for TypedHigherOrderFnInstance<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> HigherOrderFnId {
        V::id(&self.vtable)
    }

    fn options_any(&self) -> &dyn Any {
        &self.options
    }

    fn arity(&self) -> Arity {
        V::arity(&self.vtable, &self.options)
    }

    fn lambda_arity(&self) -> usize {
        V::lambda_arity(&self.vtable, &self.options)
    }

    fn child_name(&self, child_idx: usize) -> ChildName {
        V::child_name(&self.vtable, &self.options, child_idx)
    }

    fn options_serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        V::serialize(&self.vtable, &self.options)
    }

    fn options_eq(&self, other_options: &dyn Any) -> bool {
        other_options
            .downcast_ref::<V::Options>()
            .is_some_and(|options| self.options == *options)
    }

    fn options_hash(&self, mut hasher: &mut dyn Hasher) {
        std::hash::Hash::hash(&self.options, &mut hasher);
    }

    fn options_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.options, f)
    }

    fn options_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.options, f)
    }
}
