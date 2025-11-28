// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::transmute;
use std::ops::Deref;
use std::sync::Arc;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::vortex_bail;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_vector::Vector;

use crate::functions::execution::ExecutionCtx;
use crate::functions::scalar::ScalarFn;
use crate::functions::signature::Signature;
use crate::functions::FunctionId;

/// A non-object-safe vtable trait for scalar function types.
///
/// This trait should be implemented in order to define new scalar functions within Vortex.
pub trait VTable: 'static + Send + Sync {
    /// Any options for configuring the function's behaviour.
    type Options: 'static
        + Send
        + Sync
        + Default
        + Clone
        + PartialEq
        + Eq
        + Hash
        + fmt::Debug
        + fmt::Display;

    /// The globally unique identifier for this function.
    fn id(&self) -> FunctionId;

    /// Serializes the options for a function instance.
    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Deserializes the options for this function from a byte slice.
    fn deserialize(&self, _bytes: &[u8]) -> VortexResult<Self::Options> {
        vortex_bail!("Serialization is not supported for {}", self.id())
    }

    /// Returns signature information about this function.
    fn signature(&self, _options: &Self::Options) -> impl Signature;

    /// Computes the return [`DType`] given the argument types and function options.
    fn return_dtype(&self, options: &Self::Options, arg_types: &[DType]) -> VortexResult<DType>;

    /// Binds the function for execution over a specific set of inputs.
    // TODO(ngates): in the future, we should return a kernel as a node in a physical plan and
    //  continue to run further cost-based optimizations prior to execution.
    fn execute(&self, _options: &Self::Options, _ctx: &ExecutionCtx) -> VortexResult<Vector> {
        vortex_bail!("Execution is not supported for {}", self.id())
    }
}

/// An object-safe vtable for scalar functions that dispatches to the non-object-safe vtable.
pub(super) trait DynScalarFnVTable: 'static + Send + Sync {
    fn id(&self) -> FunctionId;

    fn serialize_options(&self, options: &dyn Any) -> VortexResult<Option<Vec<u8>>>;
    fn deserialize_options(&self, data: &[u8]) -> VortexResult<Box<dyn Any + Send + Sync>>;
    fn clone_options(&self, options: &dyn Any) -> Box<dyn Any + Send + Sync>;
    fn eq_options(&self, a: &dyn Any, b: &dyn Any) -> bool;
    fn hash_options(&self, options: &dyn Any, hasher: &mut dyn Hasher);
    fn fmt_options(&self, options: &dyn Any, fmt: &mut Formatter<'_>) -> fmt::Result;
    fn debug_options(&self, options: &dyn Any, fmt: &mut Formatter<'_>) -> fmt::Result;

    fn return_dtype(&self, options: &dyn Any, arg_types: &[DType]) -> VortexResult<DType>;
    fn execute(&self, options: &dyn Any, ctx: &ExecutionCtx) -> VortexResult<Vector>;
}

#[repr(transparent)]
pub struct ScalarFnVTableAdapter<V>(V);
impl<V: VTable> DynScalarFnVTable for ScalarFnVTableAdapter<V> {
    fn id(&self) -> FunctionId {
        V::id(&self.0)
    }

    fn serialize_options(&self, options: &dyn Any) -> VortexResult<Option<Vec<u8>>> {
        V::serialize(&self.0, downcast::<V>(options))
    }

    fn deserialize_options(&self, data: &[u8]) -> VortexResult<Box<dyn Any + Send + Sync>> {
        Ok(Box::new(V::deserialize(&self.0, data)?))
    }

    fn clone_options(&self, options: &dyn Any) -> Box<dyn Any + Send + Sync> {
        Box::new(downcast::<V>(options).clone())
    }

    fn eq_options(&self, a: &dyn Any, b: &dyn Any) -> bool {
        downcast::<V>(a) == downcast::<V>(b)
    }

    fn hash_options(&self, options: &dyn Any, mut hasher: &mut dyn Hasher) {
        downcast::<V>(options).hash(&mut hasher);
    }

    fn fmt_options(&self, options: &dyn Any, f: &mut Formatter) -> fmt::Result {
        fmt::Display::fmt(downcast::<V>(options), f)
    }

    fn debug_options(&self, options: &dyn Any, f: &mut Formatter) -> fmt::Result {
        fmt::Debug::fmt(downcast::<V>(options), f)
    }

    fn return_dtype(&self, options: &dyn Any, arg_types: &[DType]) -> VortexResult<DType> {
        V::return_dtype(&self.0, downcast::<V>(options), arg_types)
    }

    fn execute(&self, options: &dyn Any, ctx: &ExecutionCtx) -> VortexResult<Vector> {
        // TODO(ngates): validate result matches expected dtype from ctx.
        V::execute(&self.0, downcast::<V>(options), ctx)
    }
}

fn downcast<V: VTable>(options: &dyn Any) -> &V::Options {
    options
        .downcast_ref::<V::Options>()
        .vortex_expect("Invalid options type for scalar function")
}

/// A vtable for scalar functions, registered against a VortexSession.
#[derive(Clone)]
pub struct ScalarFnVTable(ArcRef<dyn DynScalarFnVTable>);

impl ScalarFnVTable {
    /// Creates a ScalarFnVTable from a VTable implementation.
    pub fn new<F: VTable>(vtable: F) -> Self {
        Self(ArcRef::new_arc(Arc::new(ScalarFnVTableAdapter(vtable))))
    }

    /// Creates a ScalarFnVTable from a 'static reference to a VTable.
    pub fn new_static<F: VTable>(vtable: &'static F) -> Self {
        // SAFETY: this transmute is safe since ScalarFnVTableAdapter is transparent over F.
        let adapter: &'static ScalarFnVTableAdapter<F> =
            unsafe { transmute::<&'static F, &'static ScalarFnVTableAdapter<F>>(vtable) };
        Self(ArcRef::new_ref(adapter))
    }

    /// Crate-local function for accessing the underlying vtable.
    pub(super) fn as_dyn(&self) -> &dyn DynScalarFnVTable {
        self.0.deref()
    }

    pub fn id(&self) -> FunctionId {
        self.0.id()
    }

    pub fn deserialize(&self, bytes: &[u8]) -> VortexResult<ScalarFn> {
        let options = self.0.deserialize_options(bytes)?;
        // SAFETY: options were created by this vtable.
        Ok(unsafe { ScalarFn::new_unchecked(self.clone(), options) })
    }
}

impl fmt::Debug for ScalarFnVTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScalarFnVTable")
            .field("id", &self.id())
            .finish()
    }
}

/// An empty options type for functions that do not require any configuration.
#[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EmptyOptions;
impl fmt::Display for EmptyOptions {
    fn fmt(&self, _f: &mut Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}
