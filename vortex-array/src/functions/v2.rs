// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::functions::{ExecutionCtx, FunctionId, Monotonicity, NullHandling, Signature};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_utils::dyn_traits::{DynEq, DynHash};
use vortex_vector::{Scalar, Vector};

/// A reference-counted pointer to a scalar function.
pub type ScalarFnRef = Arc<dyn ScalarFn>;

/// An instance of a scalar function, including any options required for its execution.
pub trait ScalarFn: 'static + Send + Sync + Debug + DynEq + DynHash {
    /// Returns the unique identifier for this scalar function.
    fn id(&self) -> FunctionId;

    /// Returns signature information about this function.
    fn signature(&self) -> &dyn Signature;

    /// Computes the return [`DType`] given the argument types and function options.
    fn return_dtype(&self, arg_types: &[DType]) -> VortexResult<DType>;

    /// Binds the function for execution over a specific set of inputs.
    // TODO(ngates): in the future, we should return a kernel as a node in a physical plan and
    //  continue to run further cost-based optimizations prior to execution.
    fn execute(&self, _ctx: &ExecutionCtx) -> VortexResult<Vector> {
        vortex_bail!("Execution is not supported for {}", self.id())
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
    fn identity_element(&self) -> Option<Scalar> {
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
    fn absorbing_element(&self) -> Option<Scalar> {
        None
    }

    /// The number of arguments accepted by this function.
    fn arity(&self) -> usize {
        self.signature().arity()
    }

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
    ///            isotone in y (larger y → more likely that x < y)
    /// - `x + y`: isotone in both arguments
    /// - `x - y`: isotone in x, antitone in y
    /// - `ABS(x)`: neither (non-monotonic)
    fn monotonicity(&self, arg_idx: usize) -> Monotonicity {
        _ = arg_idx;
        Monotonicity::default()
    }

    /// Whether argument order is irrelevant: `f(a, b) = f(b, a)`.
    ///
    /// Enables expression normalization (e.g., sorting arguments by column id)
    /// for better common subexpression elimination and pattern matching.
    ///
    /// # Examples
    /// - Commutative: `+`, `*`, `AND`, `OR`, `=`, `!=`, `MIN`, `MAX`
    /// - Non-commutative: `-`, `/`, `<`, `>`, `CONCAT`
    fn is_commutative(&self) -> bool {
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
    fn is_idempotent(&self) -> bool {
        false
    }

    /// Whether `f(f(x)) = x` for unary functions.
    ///
    /// Enables cancellation of nested self-applications.
    ///
    /// # Examples
    /// - Involutions: `NOT`, `NEG` (for signed types), `REVERSE`
    /// - Non-involutions: `ABS`, `UPPER`, `LOWER`
    fn is_involution(&self) -> bool {
        false
    }

    /// How the function behaves when one or more arguments are NULL.
    ///
    /// Most functions propagate NULL (any NULL argument produces NULL output).
    /// Some functions have special NULL handling that can short-circuit
    /// evaluation or treat NULL as a meaningful value.
    ///
    /// Required for correct NULL semantics; may also enable optimizations
    /// when argument nullability is known from schema or statistics.
    fn null_handling(&self) -> NullHandling {
        NullHandling::default()
    }
}

impl Hash for dyn ScalarFn + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}
impl PartialEq for dyn ScalarFn + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}
impl Eq for dyn ScalarFn + '_ {}

/// A reference-counted pointer to a scalar function codec.
pub type ScalarFnCodecRef = Arc<dyn ScalarFnCodec>;

/// A codec for serializing and deserializing scalar functions.
pub trait ScalarFnCodec: 'static + Send + Sync + Debug {
    /// Serialize the given scalar function into a byte vector.
    ///
    /// The `id` of the function should not be serialized as part of this method.
    ///
    /// If the function does not support serialization, return `Ok(None)`.
    fn serialize(&self, function: &dyn ScalarFn) -> VortexResult<Option<Vec<u8>>> {
        _ = function;
        Ok(None)
    }

    /// Deserialize a scalar function from the given byte slice.
    fn deserialize(&self, bytes: &[u8]) -> VortexResult<ScalarFnRef>;
}
