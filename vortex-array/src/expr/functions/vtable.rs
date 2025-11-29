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
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Datum;

use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::functions::ArgName;
use crate::expr::functions::FunctionId;
use crate::expr::functions::execution::ExecutionCtx;
use crate::expr::functions::scalar::ScalarFn;
use crate::expr::stats::Stat;

/// A non-object-safe vtable trait for scalar function types.
///
/// This trait should be implemented in order to define new scalar functions within Vortex.
pub trait VTable: 'static + Send + Sync + Sized {
    /// Any options for configuring the function's behaviour.
    type Options: 'static + Send + Sync + Clone + PartialEq + Eq + Hash + fmt::Debug + fmt::Display;

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

    /// Returns the arity (number of arguments) for this function.
    fn arity(&self, options: &Self::Options) -> Arity;

    /// How the function behaves when one or more arguments are NULL.
    ///
    /// Most functions propagate NULL (any NULL argument produces NULL output).
    /// Some functions have special NULL handling that can short-circuit
    /// evaluation or treat NULL as a meaningful value.
    ///
    /// Required for correct NULL semantics; may also enable optimizations
    /// when argument nullability is known from schema or statistics.
    fn null_handling(&self, options: &Self::Options) -> NullHandling {
        _ = options;
        NullHandling::default()
    }

    /// Returns the display name of the nth argument for this function.
    fn arg_name(&self, options: &Self::Options, arg_idx: usize) -> ArgName;

    /// See [`Expression::stat_falsification`]
    ///
    /// Note that the falsification API will change in the future to instead use a `falsify`
    /// expression along with push-down rules.
    fn stat_falsification(
        &self,
        options: &Self::Options,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        _ = options;
        _ = expr;
        _ = catalog;
        None
    }

    /// See [`Expression::stat_expression`]
    ///
    /// Note that the stat_expression API will change in the future such that layouts with pruning
    /// capabilities perform their own mapping over statistics.
    fn stat_expression(
        &self,
        options: &Self::Options,
        expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        _ = options;
        _ = expr;
        _ = stat;
        _ = catalog;
        None
    }

    /// Computes the return [`DType`] given the argument types and function options.
    fn return_dtype(&self, options: &Self::Options, arg_types: &[DType]) -> VortexResult<DType>;

    /// Binds the function for execution over a specific set of inputs.
    // TODO(ngates): in the future, we should return a kernel as a node in a physical plan and
    //  continue to run further cost-based optimizations prior to execution.
    fn execute(&self, _options: &Self::Options, _ctx: &ExecutionCtx) -> VortexResult<Datum>;
}

/// The arity (number of arguments) of a function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arity {
    Fixed(usize),
    Variadic { min: usize, max: Option<usize> },
}

impl Arity {
    /// Whether the given argument count matches this arity.
    pub fn matches(&self, arg_count: usize) -> bool {
        match self {
            Arity::Fixed(m) => *m == arg_count,
            Arity::Variadic { min, max } => {
                if arg_count < *min {
                    return false;
                }
                if let Some(max) = max
                    && arg_count > *max
                {
                    return false;
                }
                true
            }
        }
    }
}

/// How a function handles NULL arguments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum NullHandling {
    /// NULL in any argument produces NULL output.
    ///
    /// This is standard SQL behavior for most scalar functions.
    /// Enables simplification when any argument is known to be NULL.
    Propagate,

    /// NULL is short-circuited when paired with the absorbing element.
    ///
    /// This is a special case where the absorbing element "wins" over NULL.
    ///
    /// # Examples
    /// - `AND_KLEENE(false, NULL)` → `false` (false absorbs NULL)
    /// - `OR_KLEENE(true, NULL)` → `true` (true absorbs NULL)
    AbsorbsNull,

    /// The function has special NULL semantics that don't follow
    /// simple propagation rules.
    ///
    /// This prevents any simplifications based on NULL arguments.
    ///
    /// # Examples
    /// - `IS NULL`, `IS NOT NULL`: NULL → true/false
    /// - `COALESCE`: returns first non-NULL argument
    /// - `NULLIF`: conditionally produces NULL
    #[default]
    Custom,
}

/// An object-safe vtable for scalar functions that dispatches to the non-object-safe vtable.
pub(crate) trait DynScalarFnVTable: 'static + Send + Sync {
    fn id(&self) -> FunctionId;

    fn options_serialize(&self, options: &dyn Any) -> VortexResult<Option<Vec<u8>>>;
    fn options_deserialize(&self, data: &[u8]) -> VortexResult<Box<dyn Any + Send + Sync>>;
    fn options_clone(&self, options: &dyn Any) -> Box<dyn Any + Send + Sync>;
    fn options_eq(&self, a: &dyn Any, b: &dyn Any) -> bool;
    fn options_hash(&self, options: &dyn Any, hasher: &mut dyn Hasher);
    fn options_display(&self, options: &dyn Any, fmt: &mut Formatter<'_>) -> fmt::Result;
    fn options_debug(&self, options: &dyn Any, fmt: &mut Formatter<'_>) -> fmt::Result;

    fn arity(&self, options: &dyn Any) -> Arity;
    fn arg_name(&self, options: &dyn Any, arg_idx: usize) -> ArgName;
    fn null_handling(&self, options: &dyn Any) -> NullHandling;

    fn stat_falsification(
        &self,
        options: &dyn Any,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression>;
    fn stat_expression(
        &self,
        options: &dyn Any,
        expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression>;

    fn return_dtype(&self, options: &dyn Any, arg_types: &[DType]) -> VortexResult<DType>;
    fn execute(&self, options: &dyn Any, ctx: &ExecutionCtx) -> VortexResult<Datum>;
}

#[repr(transparent)]
pub struct ScalarFnVTableAdapter<V>(V);
impl<V: VTable> DynScalarFnVTable for ScalarFnVTableAdapter<V> {
    fn id(&self) -> FunctionId {
        V::id(&self.0)
    }

    fn options_serialize(&self, options: &dyn Any) -> VortexResult<Option<Vec<u8>>> {
        V::serialize(&self.0, downcast::<V>(options))
    }

    fn options_deserialize(&self, data: &[u8]) -> VortexResult<Box<dyn Any + Send + Sync>> {
        Ok(Box::new(V::deserialize(&self.0, data)?))
    }

    fn options_clone(&self, options: &dyn Any) -> Box<dyn Any + Send + Sync> {
        Box::new(downcast::<V>(options).clone())
    }

    fn options_eq(&self, a: &dyn Any, b: &dyn Any) -> bool {
        downcast::<V>(a) == downcast::<V>(b)
    }

    fn options_hash(&self, options: &dyn Any, mut hasher: &mut dyn Hasher) {
        downcast::<V>(options).hash(&mut hasher);
    }

    fn options_display(&self, options: &dyn Any, f: &mut Formatter) -> fmt::Result {
        fmt::Display::fmt(downcast::<V>(options), f)
    }

    fn options_debug(&self, options: &dyn Any, f: &mut Formatter) -> fmt::Result {
        fmt::Debug::fmt(downcast::<V>(options), f)
    }

    fn arity(&self, options: &dyn Any) -> Arity {
        V::arity(&self.0, downcast::<V>(options))
    }

    fn arg_name(&self, options: &dyn Any, arg_idx: usize) -> ArgName {
        V::arg_name(&self.0, downcast::<V>(options), arg_idx)
    }

    fn null_handling(&self, options: &dyn Any) -> NullHandling {
        V::null_handling(&self.0, downcast::<V>(options))
    }

    fn stat_falsification(
        &self,
        options: &dyn Any,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        V::stat_falsification(&self.0, downcast::<V>(options), expr, catalog)
    }

    fn stat_expression(
        &self,
        options: &dyn Any,
        expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        V::stat_expression(&self.0, downcast::<V>(options), expr, stat, catalog)
    }

    fn return_dtype(&self, options: &dyn Any, arg_types: &[DType]) -> VortexResult<DType> {
        V::return_dtype(&self.0, downcast::<V>(options), arg_types)
    }

    fn execute(&self, options: &dyn Any, ctx: &ExecutionCtx) -> VortexResult<Datum> {
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
    pub(crate) fn as_dyn(&self) -> &dyn DynScalarFnVTable {
        self.0.deref()
    }

    pub fn id(&self) -> FunctionId {
        self.0.id()
    }

    pub fn deserialize(&self, bytes: &[u8]) -> VortexResult<ScalarFn> {
        let options = self.0.options_deserialize(bytes)?;
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

pub trait ScalarFnVTableExt: VTable {
    /// Creates a new ScalarFn instance with the given options.
    fn new(self, options: Self::Options) -> ScalarFn {
        ScalarFn::new(self, options)
    }

    /// Creates a new ScalarFn instance with the given options from a 'static vtable.
    fn new_static(&'static self, options: Self::Options) -> ScalarFn {
        ScalarFn::new_static(self, options)
    }
}
impl<V: VTable> ScalarFnVTableExt for V {}
