// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use arcref::ArcRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::stats::Stat;
use crate::expr::traversal::Node;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::TypedScalarFnInstance;

/// This trait defines the interface for scalar function vtables, including methods for
/// serialization, deserialization, validation, child naming, return type computation,
/// and evaluation.
///
/// This trait is non-object safe and allows the implementer to make use of associated types
/// for improved type safety, while allowing Vortex to enforce runtime checks on the inputs and
/// outputs of each function.
///
/// The [`ScalarFnVTable`] trait should be implemented for a struct that holds global data across
/// all instances of the expression. In almost all cases, this struct will be an empty unit
/// struct, since most expressions do not require any global state.
pub trait ScalarFnVTable: 'static + Sized + Clone + Send + Sync {
    /// Options for this expression.
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// Returns the ID of the scalar function vtable.
    fn id(&self) -> ScalarFnId;

    /// Serialize the options for this expression.
    ///
    /// Should return `Ok(None)` if the expression is not serializable, and `Ok(vec![])` if it is
    /// serializable but has no metadata.
    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        _ = options;
        Ok(None)
    }

    /// Deserialize the options of this expression.
    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_bail!("Expression {} is not deserializable", self.id());
    }

    /// Returns the arity of this expression.
    fn arity(&self, options: &Self::Options) -> Arity;

    /// Returns the name of the nth child of the expr.
    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName;

    /// Format this expression in a nice human-readable SQL-style format
    ///
    /// The implementation should recursively format child expressions by calling
    /// `expr.child(i).fmt_sql(f)`.
    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}(", self.id())?;
        let nchildren = expr.children_count();
        for (i, child) in expr.children().iter().enumerate() {
            child.fmt_sql(f)?;
            if i + 1 < nchildren {
                write!(f, ", ")?;
            }
        }
        let opts = format!("{}", options);
        if !opts.is_empty() {
            write!(f, ", opts={}", opts)?;
        }
        write!(f, ")")
    }

    /// Compute the return [`DType`] of the expression if evaluated over the given input types.
    ///
    /// # Preconditions
    ///
    /// The length of `args` must match the [`Arity`] of this function. Callers are responsible
    /// for validating this (e.g., [`Expression::try_new`] checks arity at construction time).
    /// Implementations may assume correct arity and will panic or return nonsensical results if
    /// violated.
    ///
    /// [`Expression::try_new`]: crate::expr::Expression::try_new
    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType>;

    /// Execute the expression over the input arguments.
    ///
    /// Implementations are encouraged to check their inputs for constant arrays to perform
    /// more optimized execution.
    ///
    /// If the input arguments cannot be directly used for execution (for example, an expression
    /// may require canonical input arrays), then the implementation should perform a single
    /// child execution and return a new [`crate::arrays::ScalarFnArray`] wrapping up the new child.
    ///
    /// This provides maximum opportunities for array-level optimizations using execute_parent
    /// kernels.
    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef>;

    /// Implement an abstract reduction rule over a tree of scalar functions.
    ///
    /// The [`ReduceNode`] can be used to traverse children, inspect their types, and
    /// construct the result expression.
    ///
    /// Return `Ok(None)` if no reduction is possible.
    fn reduce(
        &self,
        options: &Self::Options,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        _ = options;
        _ = node;
        _ = ctx;
        Ok(None)
    }

    /// Simplify the expression if possible.
    fn simplify(
        &self,
        options: &Self::Options,
        expr: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        _ = options;
        _ = expr;
        _ = ctx;
        Ok(None)
    }

    /// Simplify the expression if possible, without type information.
    fn simplify_untyped(
        &self,
        options: &Self::Options,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        _ = options;
        _ = expr;
        Ok(None)
    }

    /// See [`Expression::stat_falsification`].
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

    /// See [`Expression::stat_expression`].
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

    /// Returns an expression that evaluates to the validity of the result of this expression.
    ///
    /// If a validity expression cannot be constructed, returns `None` and the expression will
    /// be evaluated as normal before extracting the validity mask from the result.
    ///
    /// This is essentially a specialized form of a `reduce_parent`
    fn validity(
        &self,
        options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        _ = (options, expression);
        Ok(None)
    }

    /// Returns whether this expression itself is null-sensitive. Conservatively default to *true*.
    ///
    /// An expression is null-sensitive if it directly operates on null values,
    /// such as `is_null`. Most expressions are not null-sensitive.
    ///
    /// The property we are interested in is if the expression (e) distributes over `mask`.
    /// Define a `mask(a, m)` expression that applies the boolean array `m` to the validity of the
    /// array `a`.
    ///
    /// A unary expression `e` is not null-sensitive iff forall arrays `a` and masks `m`,
    /// `e(mask(a, m)) == mask(e(a), m)`.
    ///
    /// This can be extended to an n-ary expression.
    ///
    /// This method only checks the expression itself, not its children.
    fn is_null_sensitive(&self, options: &Self::Options) -> bool {
        _ = options;
        true
    }

    /// Returns whether this expression is semantically fallible. Conservatively defaults to
    /// `true`.
    ///
    /// An expression is semantically fallible if there exists a set of well-typed inputs that
    /// causes the expression to produce an error as part of its _defined behavior_. For example,
    /// `checked_add` is fallible because integer overflow is a domain error, and division is
    /// fallible because of division by zero.
    ///
    /// This does **not** include execution errors that are incidental to the implementation, such
    /// as canonicalization failures, memory allocation errors, or encoding mismatches. Those can
    /// happen to any expression and are not what this method captures.
    ///
    /// This property is used by optimizations that speculatively evaluate an expression over values
    /// that may not appear in the actual input. For example, pushing a scalar function down to a
    /// dictionary's values array is only safe when the function is infallible or all values are
    /// referenced, since a fallible function might error on a value left unreferenced after
    /// slicing that would never be encountered during normal evaluation.
    ///
    /// Note: this is only applicable to expressions that pass type-checking via
    /// [`ScalarFnVTable::return_dtype`].
    fn is_fallible(&self, options: &Self::Options) -> bool {
        _ = options;
        true
    }
}

/// Arguments for reduction rules.
pub trait ReduceCtx {
    /// Create a new reduction node from the given scalar function and children.
    fn new_node(
        &self,
        scalar_fn: ScalarFnRef,
        children: &[ReduceNodeRef],
    ) -> VortexResult<ReduceNodeRef>;
}

pub type ReduceNodeRef = Arc<dyn ReduceNode>;

/// A node used for implementing abstract reduction rules.
pub trait ReduceNode {
    /// Downcast to Any.
    fn as_any(&self) -> &dyn Any;

    /// Return the data type of this node.
    fn node_dtype(&self) -> VortexResult<DType>;

    /// Return this node's scalar function if it is indeed a scalar fn.
    fn scalar_fn(&self) -> Option<&ScalarFnRef>;

    /// Descend to the child of this handle.
    fn child(&self, idx: usize) -> ReduceNodeRef;

    /// Returns the number of children of this node.
    fn child_count(&self) -> usize;
}

/// The arity (number of arguments) of a function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arity {
    Exact(usize),
    Variadic { min: usize, max: Option<usize> },
}

impl Display for Arity {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Arity::Exact(n) => write!(f, "{}", n),
            Arity::Variadic { min, max } => match max {
                Some(max) if min == max => write!(f, "{}", min),
                Some(max) => write!(f, "{}..{}", min, max),
                None => write!(f, "{}+", min),
            },
        }
    }
}

impl Arity {
    /// Whether the given argument count matches this arity.
    pub fn matches(&self, arg_count: usize) -> bool {
        match self {
            Arity::Exact(m) => *m == arg_count,
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

/// Context for simplification.
///
/// Used to lazily compute input data types where simplification requires them.
pub trait SimplifyCtx {
    /// Get the data type of the given expression.
    fn return_dtype(&self, expr: &Expression) -> VortexResult<DType>;
}

/// Arguments for expression execution.
pub trait ExecutionArgs {
    /// Returns the input array at the given index.
    fn get(&self, index: usize) -> VortexResult<ArrayRef>;

    /// Returns the number of inputs.
    fn num_inputs(&self) -> usize;

    /// Returns the row count of the execution scope.
    fn row_count(&self) -> usize;
}

/// A concrete [`ExecutionArgs`] backed by a `Vec<ArrayRef>`.
pub struct VecExecutionArgs {
    inputs: Vec<ArrayRef>,
    row_count: usize,
}

impl VecExecutionArgs {
    /// Create a new `VecExecutionArgs`.
    pub fn new(inputs: Vec<ArrayRef>, row_count: usize) -> Self {
        Self { inputs, row_count }
    }
}

impl ExecutionArgs for VecExecutionArgs {
    fn get(&self, index: usize) -> VortexResult<ArrayRef> {
        self.inputs.get(index).cloned().ok_or_else(|| {
            vortex_err!(
                "Input index {} out of bounds (num_inputs={})",
                index,
                self.inputs.len()
            )
        })
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    fn row_count(&self) -> usize {
        self.row_count
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EmptyOptions;
impl Display for EmptyOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

/// Factory functions for vtables.
pub trait ScalarFnVTableExt: ScalarFnVTable {
    /// Bind this vtable with the given options into a [`ScalarFnRef`].
    fn bind(&self, options: Self::Options) -> ScalarFnRef {
        TypedScalarFnInstance::new(self.clone(), options).erased()
    }

    /// Create a new expression with this vtable and the given options and children.
    fn new_expr(
        &self,
        options: Self::Options,
        children: impl IntoIterator<Item = Expression>,
    ) -> Expression {
        Self::try_new_expr(self, options, children).vortex_expect("Failed to create expression")
    }

    /// Try to create a new expression with this vtable and the given options and children.
    fn try_new_expr(
        &self,
        options: Self::Options,
        children: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Expression> {
        Expression::try_new(self.bind(options), children)
    }
}
impl<V: ScalarFnVTable> ScalarFnVTableExt for V {}

/// A reference to the name of a child expression.
pub type ChildName = ArcRef<str>;
