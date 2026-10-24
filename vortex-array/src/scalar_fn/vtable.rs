// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::borrow::Cow;
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
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::TypedScalarFnInstance;
use crate::scalar_fn::builtin::binary::Binary;
use crate::scalar_fn::builtin::is_not_null::IsNotNull;
use crate::scalar_fn::builtin::operators::Operator;
use crate::scalar_fn::display::ExprDisplay;
use crate::validity::Validity;

// Here and beyond we use validity(x) interchangeably with is_not_null(x).
#[derive(Clone)]
pub enum ReduceNodeValidity<T: ReduceNode> {
    /// Validity of T can be symbolically reduced (i.e. without evaluating T)
    /// to a function over validities of this node's children. For an
    /// expression reduce node, one example is byte_length(x).
    /// validity(byte_length(x)) can be symbolically reduced to validity(x)
    /// since byte_length doesn't change validity.
    Reduced(T),
    /// Validity of T can't be symbolically reduced to anything, and all
    /// further reductions require evaluating T first. For an expression reduce
    /// node, one example is list_contains(x, C) since you can't reduce
    /// validity(list_contains(x, C)). list_contains([], null) is false, but to
    /// know that, you need to evaluate x's offsets.
    ///
    /// This is also the default case.
    Irreducible,
}

/// IsNotNull(child) as a reducible node
pub fn is_not_null_node<T: ReduceNode>(child: &T) -> VortexResult<T> {
    child.new_node(IsNotNull.bind(EmptyOptions), std::slice::from_ref(child))
}

/// "And" over "node's" non-nullable children
pub fn union_child_validities<T: ReduceNode>(node: &T) -> VortexResult<T> {
    let mut parts = Vec::with_capacity(node.child_count());
    for i in 0..node.child_count() {
        let child = node.child(i);
        if child.node_dtype()?.is_nullable() {
            parts.push(is_not_null_node(&child)?);
        }
    }
    let parts = parts
        .into_iter()
        .try_reduce_balanced(|lhs, rhs| {
            lhs.new_node(Binary.bind(Operator::And), &[lhs.clone(), rhs])
        })?
        .unwrap_or_else(|| node.new_constant(true.into()));
    Ok(parts)
}

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

    /// Format an expression tree in a human-readable SQL-style format.
    ///
    /// The expression may be either an unbound or a bound expression.
    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}(", self.id())?;
        let nchildren = expr.display_children_count();
        for i in 0..nchildren {
            Display::fmt(expr.display_child(i), f)?;
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
    /// for validating this (e.g., `Expression::try_new` checks arity at construction time).
    /// Implementations may assume correct arity and will panic or return nonsensical results if
    /// violated.
    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType>;

    /// Bind whatever in the options depends on the argument dtypes, such as the body of a
    /// `Lambda`, returning the bound options.
    ///
    /// This is called whenever a bound node is built and whenever a
    /// [`ScalarFnArray`] is built from its children, before
    /// [`return_dtype`](Self::return_dtype), so [`execute`](Self::execute) finds the options bound.
    /// A rebuilt node calls it again, possibly with other dtypes: options that are already bound
    /// must be checked against these dtypes, returning `None` if they fit and an error if they
    /// don't. Return `None` to keep the options as they are, which the default does.
    #[doc(hidden)]
    fn bind_options(
        &self,
        options: &Self::Options,
        args: &[DType],
    ) -> VortexResult<Option<Self::Options>> {
        _ = (options, args);
        Ok(None)
    }

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
    /// construct the result via [`ReduceNode::new_node`]. The rule is generic over the node
    /// type and is instantiated once per reducible tree kind (expressions and arrays).
    ///
    /// Return `Ok(None)` if no reduction is possible.
    fn reduce<T: ReduceNode>(&self, options: &Self::Options, node: &T) -> VortexResult<Option<T>> {
        _ = options;
        _ = node;
        Ok(None)
    }

    /// For node, returns node' which is exactly the result of evaluating
    /// validity(node). Returned node' is either a lazy computation over
    /// children of node, a constant, or Irreducible which means you need to
    /// evaluate node to get its validity.
    fn validity<T: ReduceNode>(
        &self,
        options: &Self::Options,
        node: &T,
    ) -> VortexResult<ReduceNodeValidity<T>> {
        if !self.is_strict(options) {
            return Ok(ReduceNodeValidity::Irreducible);
        }

        let mut dtypes = Vec::with_capacity(node.child_count());
        for i in 0..node.child_count() {
            let dtype = node.child(i).node_dtype()?;
            if matches!(dtype, DType::Null) {
                return Ok(ReduceNodeValidity::Irreducible);
            }
            dtypes.push(dtype.as_nonnullable());
        }

        let res = if let Ok(dtype) = self.return_dtype(options, &dtypes)
            && !dtype.is_nullable()
        {
            ReduceNodeValidity::Reduced(union_child_validities(node)?)
        } else {
            ReduceNodeValidity::Irreducible
        };
        Ok(res)
    }

    /// Simplify an expression node if possible.
    ///
    /// Unlike [`ScalarFnVTable::reduce`], which the array optimizer also applies and which must
    /// therefore keep the node's exact dtype, a simplification is applied by the expression
    /// optimizer only and may rewrite the node into one whose dtype differs in nullability.
    /// Every node of `node`'s tree carries its dtype, so rules read types directly from it.
    fn simplify<T: ReduceNode>(
        &self,
        options: &Self::Options,
        node: &T,
    ) -> VortexResult<Option<T>> {
        _ = options;
        _ = node;
        Ok(None)
    }

    /// Returns whether this scalar function is strict.
    ///
    /// A strict function returns null for a row when any argument is null for that row. This
    /// matches [PostgreSQL's `STRICT` convention](https://www.postgresql.org/docs/current/sql-createfunction.html)
    /// for null propagation.
    ///
    /// Return `true` only when this holds for every argument. `add` is strict, but Kleene `AND`
    /// is not because `false AND null` returns `false`. `is_null` is also not strict.
    ///
    /// Strictness does not require valid inputs to produce a valid output. For example,
    /// `list_sum` returns null for a valid empty list. Implement
    /// [`ScalarFnVTable::validity`] only when the output validity can be derived without
    /// evaluation.
    ///
    /// [`ScalarFnVTable::return_dtype`] must return a nullable output dtype when any input dtype is
    /// nullable. A `cast` that forces a non-nullable output dtype is therefore not strict.
    ///
    /// This property applies only to the scalar function, not its child expressions. Nullary
    /// functions are vacuously strict. The default is conservatively `false`.
    fn is_strict(&self, options: &Self::Options) -> bool {
        _ = options;
        false
    }

    /// Returns whether this scalar function can never raise a semantic error.
    ///
    /// Return `true` only when a well-typed call cannot error because of its values. `checked_add`
    /// is fallible on integer overflow, and integer division is fallible when its divisor is zero.
    /// A null result is not an error: `list_sum` is infallible for an empty list.
    ///
    /// Ignore incidental execution errors, such as canonicalization failures, allocation errors,
    /// and encoding mismatches. They are not part of the function's semantics.
    ///
    /// Returning `true` permits optimizations that evaluate the function over values that no input
    /// row references. Dictionary push-down, for example, evaluates every dictionary value, so a
    /// fallible function could error on a value that row-wise evaluation would never reach.
    ///
    /// This applies only to the scalar function, not its child expressions, and only to inputs
    /// accepted by [`ScalarFnVTable::return_dtype`]. The default is conservatively `false`.
    fn is_infallible(&self, options: &Self::Options) -> bool {
        _ = options;
        false
    }
}

/// A node used for implementing abstract reduction rules over a tree of scalar functions.
///
/// Reduction rules are generic over the node type, so a rule is written once and monomorphized
/// per reducible tree kind: [`ArrayReduceNode`] for array trees and [`DynReduceNode`] for tree
/// kinds defined outside this crate, such as bound expressions. Nodes borrow from the tree being reduced, making
/// traversal allocation-free, while nodes produced by [`ReduceNode::new_node`] own their
/// freshly-built subtrees.
pub trait ReduceNode: Clone {
    /// Return the data type of this node.
    fn node_dtype(&self) -> VortexResult<DType>;

    /// Return this node's scalar function if it is indeed a scalar fn.
    fn scalar_fn(&self) -> Option<&ScalarFnRef>;

    /// Descend to the child of this node.
    fn child(&self, idx: usize) -> Self;

    /// Returns the number of children of this node.
    fn child_count(&self) -> usize;

    /// Create a new node from the given scalar function and children, inheriting this node's
    /// reduction context (e.g. the expression scope, or the array row count).
    fn new_node(&self, scalar_fn: ScalarFnRef, children: &[Self]) -> VortexResult<Self>;

    /// Return a scalar value if this node is constant
    fn as_constant(&self) -> Option<Scalar> {
        None
    }

    /// Produce a new constant node in the same scope as "self"
    fn new_constant(&self, value: Scalar) -> Self;

    /// Symbolic validity of this node. Reduced() if you can get from node's
    /// validity to validity of its children or a constant without evaluating
    /// node.
    fn validity(&self) -> VortexResult<ReduceNodeValidity<Self>>
    where
        Self: Sized;

    /// Returns whether this node is known to evaluate to the same value as `other` for every
    /// row. `false` is always a sound answer.
    fn is_same(&self, other: &Self) -> bool;
}

/// Object-safe counterpart of [`ReduceNode`] for tree kinds defined outside this crate, such as
/// bound expressions.
///
/// [`DynReduceNode`] wraps an implementation and is itself a [`ReduceNode`], so every reduction
/// rule written against [`ReduceNode`] applies to it. Nodes created by an implementation's
/// [`ErasedReduceNode::child`], [`ErasedReduceNode::new_node`] and
/// [`ErasedReduceNode::new_constant`] must wrap the same implementation, so the tree kind can
/// recover its own type from a reduced node with [`DynReduceNode::downcast_ref`].
pub trait ErasedReduceNode: Send + Sync + 'static {
    /// Returns this node as `Any`, for [`DynReduceNode::downcast_ref`].
    fn as_any(&self) -> &dyn Any;

    /// See [`ReduceNode::node_dtype`].
    fn node_dtype(&self) -> VortexResult<DType>;

    /// See [`ReduceNode::scalar_fn`].
    fn scalar_fn(&self) -> Option<&ScalarFnRef>;

    /// See [`ReduceNode::child`].
    fn child(&self, idx: usize) -> DynReduceNode;

    /// See [`ReduceNode::child_count`].
    fn child_count(&self) -> usize;

    /// See [`ReduceNode::new_node`].
    fn new_node(
        &self,
        scalar_fn: ScalarFnRef,
        children: &[DynReduceNode],
    ) -> VortexResult<DynReduceNode>;

    /// See [`ReduceNode::as_constant`].
    fn as_constant(&self) -> Option<Scalar>;

    /// See [`ReduceNode::new_constant`].
    fn new_constant(&self, value: Scalar) -> DynReduceNode;

    /// See [`ReduceNode::validity`]. `node` is the [`DynReduceNode`] wrapping `self`, for passing
    /// to [`ScalarFnRef::validity_dyn`].
    fn validity(&self, node: &DynReduceNode) -> VortexResult<ReduceNodeValidity<DynReduceNode>>;

    /// See [`ReduceNode::is_same`].
    fn is_same(&self, other: &DynReduceNode) -> bool;
}

/// A [`ReduceNode`] over a tree kind defined outside this crate. See [`ErasedReduceNode`].
#[derive(Clone)]
pub struct DynReduceNode(Arc<dyn ErasedReduceNode>);

impl DynReduceNode {
    /// Wraps a node of an erased tree kind.
    pub fn new<N: ErasedReduceNode>(node: N) -> Self {
        Self(Arc::new(node))
    }

    /// Returns the wrapped node if it is of type `N`.
    pub fn downcast_ref<N: ErasedReduceNode>(&self) -> Option<&N> {
        self.0.as_any().downcast_ref()
    }
}

impl ReduceNode for DynReduceNode {
    fn node_dtype(&self) -> VortexResult<DType> {
        self.0.node_dtype()
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        self.0.scalar_fn()
    }

    fn child(&self, idx: usize) -> Self {
        self.0.child(idx)
    }

    fn child_count(&self) -> usize {
        self.0.child_count()
    }

    fn new_node(&self, scalar_fn: ScalarFnRef, children: &[Self]) -> VortexResult<Self> {
        self.0.new_node(scalar_fn, children)
    }

    fn as_constant(&self) -> Option<Scalar> {
        self.0.as_constant()
    }

    fn new_constant(&self, value: Scalar) -> Self {
        self.0.new_constant(value)
    }

    fn validity(&self) -> VortexResult<ReduceNodeValidity<Self>> {
        self.0.validity(self)
    }

    fn is_same(&self, other: &Self) -> bool {
        self.0.is_same(other)
    }
}

/// A [`ReduceNode`] over an array tree.
#[derive(Clone)]
pub struct ArrayReduceNode<'a> {
    array: Cow<'a, ArrayRef>,
}

impl<'a> ArrayReduceNode<'a> {
    /// Creates a node borrowing the given array.
    pub fn new(array: &'a ArrayRef) -> Self {
        Self {
            array: Cow::Borrowed(array),
        }
    }

    /// Returns the array backing this node.
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Consumes this node and returns the backing array.
    pub fn into_array(self) -> ArrayRef {
        self.array.into_owned()
    }
}

impl ReduceNode for ArrayReduceNode<'_> {
    fn node_dtype(&self) -> VortexResult<DType> {
        Ok(self.array.dtype().clone())
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        self.array
            .as_opt::<ScalarFn>()
            .map(|a| a.data().scalar_fn())
    }

    fn child(&self, idx: usize) -> Self {
        let array = match &self.array {
            Cow::Borrowed(array) => Cow::Borrowed(
                array
                    .nth_child(idx)
                    .vortex_expect("child idx out of bounds"),
            ),
            Cow::Owned(array) => Cow::Owned(
                array
                    .nth_child(idx)
                    .vortex_expect("child idx out of bounds")
                    .clone(),
            ),
        };
        Self { array }
    }

    fn child_count(&self) -> usize {
        self.array.nchildren()
    }

    fn new_node(&self, scalar_fn: ScalarFnRef, children: &[Self]) -> VortexResult<Self> {
        let array = ScalarFnArray::try_new_with_len(
            scalar_fn,
            children.iter().map(|c| c.array.as_ref().clone()),
            self.array.len(),
        )?;
        Ok(Self {
            array: Cow::Owned(array.into_array()),
        })
    }

    fn as_constant(&self) -> Option<Scalar> {
        self.array.as_constant()
    }

    fn new_constant(&self, value: Scalar) -> Self {
        let array = ConstantArray::new(value, self.array.len());
        Self {
            array: Cow::Owned(array.into_array()),
        }
    }

    fn validity(&self) -> VortexResult<ReduceNodeValidity<Self>> {
        if let Some(scalar_fn) = self.array.as_opt::<ScalarFn>() {
            return scalar_fn.data().scalar_fn().validity_array(self);
        }
        Ok(ReduceNodeValidity::Reduced(
            match self.array.validity()? {
                Validity::NonNullable | Validity::AllValid => self.new_constant(true.into()),
                Validity::AllInvalid => self.new_constant(false.into()),
                Validity::Array(array) => Self {
                    array: Cow::Owned(array),
                },
            },
        ))
    }

    fn is_same(&self, other: &Self) -> bool {
        ArrayRef::ptr_eq(&self.array, &other.array)
    }
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
}
impl<V: ScalarFnVTable> ScalarFnVTableExt for V {}

/// A reference to the name of a child expression.
pub type ChildName = ArcRef<str>;
