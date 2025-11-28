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
use vortex_vector::Scalar;

use crate::functions::FunctionId;
use crate::functions::execution::ExecutionCtx;
use crate::functions::scalar::ScalarFn;

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

    /// The identity element `e` where `f(e, x) = f(x, e) = x`.
    ///
    /// When an argument is the identity element, the function can be
    /// eliminated entirely, returning the other argument unchanged.
    ///
    /// # Examples
    /// - `AND`: `true` (AND(true, x) → x)
    /// - `OR`: `false` (OR(false, x) → x)
    /// - `+`: `0` (0 + x → x)
    /// - `*`: `1` (1 * x → x)
    /// - `COALESCE`: `NULL` (COALESCE(NULL, x) → x)
    fn identity_element(&self, options: &Self::Options) -> Option<Scalar> {
        _ = options;
        None
    }

    /// The absorbing element `a` where `f(a, x) = f(x, a) = a`.
    ///
    /// When any argument is the absorbing element, the function short-circuits
    /// immediately, returning that element without evaluating other arguments.
    /// Also known as the "annihilator" or "zero element".
    ///
    /// # Examples
    /// - `AND`: `false` (AND(false, x) → false)
    /// - `OR`: `true` (OR(true, x) → true)
    /// - `*`: `0` (0 * x → 0)
    fn absorbing_element(&self, options: &Self::Options) -> Option<Scalar> {
        _ = options;
        None
    }

    /// Whether argument order is irrelevant: `f(a, b) = f(b, a)`.
    ///
    /// Enables expression normalization (e.g., sorting arguments by column id)
    /// for better common subexpression elimination and pattern matching.
    ///
    /// # Examples
    /// - Commutative: `+`, `*`, `AND`, `OR`, `=`, `!=`, `MIN`, `MAX`
    /// - Non-commutative: `-`, `/`, `<`, `>`, `CONCAT`
    fn is_commutative(&self, options: &Self::Options) -> bool {
        _ = options;
        false
    }

    /// Whether `f(x, x) = x`.
    ///
    /// Enables simplification when the same expression appears multiple times
    /// as arguments to the function.
    ///
    /// # Examples
    /// - Idempotent: `AND`, `OR`, `MIN`, `MAX`
    /// - Non-idempotent: `+` (x + x = 2x), `*` (x * x = x²)
    fn is_idempotent(&self, options: &Self::Options) -> bool {
        _ = options;
        false
    }

    /// Whether `f(f(x)) = x` for unary functions.
    ///
    /// Enables cancellation of nested self-applications.
    ///
    /// # Examples
    /// - Involutions: `NOT`, `NEG` (for signed types), `REVERSE`
    /// - Non-involutions: `ABS`, `UPPER`, `LOWER`
    fn is_involution(&self, options: &Self::Options) -> bool {
        _ = options;
        false
    }

    /// Returns the arity (number of arguments) for this function.
    fn arity(&self, options: &Self::Options) -> Arity;

    /// Per-argument monotonicity of the function with respect to the
    /// natural ordering of the argument type.
    ///
    /// A function is isotone (order-preserving) in an argument if increasing
    /// that argument never decreases the result. It is antitone (order-reversing)
    /// if increasing the argument never increases the result.
    ///
    /// Monotonicity enables zone map / min-max index falsification: if we know
    /// `x ∈ [min, max]`, we can bound `f(x)` and potentially skip data.
    ///
    /// # Examples
    /// - `x < y`: antitone in x (larger x → less likely to be less than y),
    ///   isotone in y (larger y → more likely that x < y)
    /// - `x + y`: isotone in both arguments
    /// - `x - y`: isotone in x, antitone in y
    /// - `ABS(x)`: neither (non-monotonic)
    fn monotonicity(&self, options: &Self::Options, arg_idx: usize) -> Monotonicity {
        _ = options;
        _ = arg_idx;
        Monotonicity::default()
    }

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
    fn arg_name(&self, options: &Self::Options, arg_idx: usize) -> Option<String>;

    /// Computes the return [`DType`] given the argument types and function options.
    fn return_dtype(&self, options: &Self::Options, arg_types: &[DType]) -> VortexResult<DType>;

    /// Binds the function for execution over a specific set of inputs.
    // TODO(ngates): in the future, we should return a kernel as a node in a physical plan and
    //  continue to run further cost-based optimizations prior to execution.
    fn execute(&self, _options: &Self::Options, _ctx: &ExecutionCtx) -> VortexResult<Datum> {
        vortex_bail!("Execution is not supported for {}", self.id())
    }
}

/// The arity (number of arguments) of a function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arity {
    Fixed(usize),
    Variadic { min: usize, max: Option<usize> },
}

/// Monotonicity of a function with respect to one of its arguments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Monotonicity {
    /// Order-preserving: `x ≤ y` implies `f(x) ≤ f(y)`.
    ///
    /// For zone map falsification, an isotone argument means we use the
    /// minimum bound to establish a lower bound on the result.
    Isotone,

    /// Order-reversing: `x ≤ y` implies `f(x) ≥ f(y)`.
    ///
    /// For zone map falsification, an antitone argument means we use the
    /// maximum bound to establish a lower bound on the result.
    Antitone,

    /// No monotonic relationship exists, or it is unknown.
    ///
    /// Zone map falsification cannot use this argument for pruning.
    #[default]
    None,
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
pub(super) trait DynScalarFnVTable: 'static + Send + Sync {
    fn id(&self) -> FunctionId;

    fn options_serialize(&self, options: &dyn Any) -> VortexResult<Option<Vec<u8>>>;
    fn options_deserialize(&self, data: &[u8]) -> VortexResult<Box<dyn Any + Send + Sync>>;
    fn options_clone(&self, options: &dyn Any) -> Box<dyn Any + Send + Sync>;
    fn options_eq(&self, a: &dyn Any, b: &dyn Any) -> bool;
    fn options_hash(&self, options: &dyn Any, hasher: &mut dyn Hasher);
    fn options_display(&self, options: &dyn Any, fmt: &mut Formatter<'_>) -> fmt::Result;
    fn options_debug(&self, options: &dyn Any, fmt: &mut Formatter<'_>) -> fmt::Result;

    fn arity(&self, options: &dyn Any) -> Arity;
    fn arg_name(&self, options: &dyn Any, arg_idx: usize) -> Option<String>;
    fn identity_element(&self, options: &dyn Any) -> Option<Scalar>;
    fn absorbing_element(&self, options: &dyn Any) -> Option<Scalar>;
    fn is_commutative(&self, options: &dyn Any) -> bool;
    fn is_idempotent(&self, options: &dyn Any) -> bool;
    fn is_involution(&self, options: &dyn Any) -> bool;
    fn monotonicity(&self, options: &dyn Any, arg_idx: usize) -> Monotonicity;
    fn null_handling(&self, options: &dyn Any) -> NullHandling;

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

    fn arg_name(&self, options: &dyn Any, arg_idx: usize) -> Option<String> {
        V::arg_name(&self.0, downcast::<V>(options), arg_idx)
    }

    fn identity_element(&self, options: &dyn Any) -> Option<Scalar> {
        V::identity_element(&self.0, downcast::<V>(options))
    }

    fn absorbing_element(&self, options: &dyn Any) -> Option<Scalar> {
        V::absorbing_element(&self.0, downcast::<V>(options))
    }

    fn is_commutative(&self, options: &dyn Any) -> bool {
        V::is_commutative(&self.0, downcast::<V>(options))
    }

    fn is_idempotent(&self, options: &dyn Any) -> bool {
        V::is_idempotent(&self.0, downcast::<V>(options))
    }

    fn is_involution(&self, options: &dyn Any) -> bool {
        V::is_involution(&self.0, downcast::<V>(options))
    }

    fn monotonicity(&self, options: &dyn Any, arg_idx: usize) -> Monotonicity {
        V::monotonicity(&self.0, downcast::<V>(options), arg_idx)
    }

    fn null_handling(&self, options: &dyn Any) -> NullHandling {
        V::null_handling(&self.0, downcast::<V>(options))
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
    pub(super) fn as_dyn(&self) -> &dyn DynScalarFnVTable {
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
