// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::dtype::FieldName;
use crate::expr::display::DisplayTreeExpr;
use crate::expr::display::fmt_call;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::scalar_fn::ErasedOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;

/// An empty child slice, returned by [`Expression::children`] for childless variants.
const NO_CHILDREN: &[Expression] = &[];

/// A node in a Vortex expression tree.
///
/// Most nodes call a scalar function on child expressions. The node holds the function's id and
/// its options, not the function itself: the function is looked up in a session's registry when
/// the expression is [bound](Expression::bind), so an expression can be built without the session,
/// or the crate, that defines the functions it calls.
///
/// The other two nodes are leaves that name a value rather than compute one: [`Expression::Root`],
/// the scope itself, and [`Expression::Variable`], a parameter of an enclosing lambda. They are
/// language primitives rather than registered functions, because their dtype comes from where they
/// appear rather than from children and they are not executable. A [`ScalarFnVTable`] can answer
/// neither of those, so they are variants instead.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expression {
    /// A scalar function applied to child expressions.
    Scalar {
        /// The id of the scalar function this node calls.
        id: ScalarFnId,
        /// The options the function is called with, of its own options type.
        options: ErasedOptions,
        /// Any children of this expression.
        children: Arc<Vec<Expression>>,
    },
    /// The full scope of the expression evaluation.
    Root,
    /// A parameter of the enclosing lambda, read by name.
    ///
    /// Binding a [`Lambda`](crate::expr::Lambda) turns its variables into reads of the struct of
    /// its inputs. Anywhere else a variable fails to bind, and it is not serializable yet.
    Variable(FieldName),
}

impl Expression {
    /// Create a node calling the scalar function registered under `id` with `options`.
    ///
    /// Nothing is checked here. The function is looked up, and `options` and the number of
    /// children are checked against it, when the expression is [bound](Expression::bind). Use
    /// [`ScalarFnVTableExt::try_new_expr`](crate::scalar_fn::ScalarFnVTableExt::try_new_expr) when
    /// the vtable is at hand, which checks the arity up front.
    pub fn new(
        id: ScalarFnId,
        options: ErasedOptions,
        children: impl IntoIterator<Item = Expression>,
    ) -> Self {
        Self::Scalar {
            id,
            options,
            children: Arc::new(Vec::from_iter(children)),
        }
    }

    /// Create a node calling `scalar_fn`, a function already built with its options, on
    /// `children`.
    ///
    /// The node keeps the function's id and options. Binding it looks the function up again in
    /// the session the expression is bound against.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of children does not match the function's arity.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);

        vortex_ensure!(
            scalar_fn.signature().arity().matches(children.len()),
            "Expression arity mismatch: expected {} children but got {}",
            scalar_fn.signature().arity(),
            children.len()
        );

        Ok(Self::new(
            scalar_fn.id(),
            scalar_fn.options().to_erased(),
            children,
        ))
    }

    /// Whether this expression is the scope root.
    pub fn is_root(&self) -> bool {
        matches!(self, Self::Root)
    }

    /// The id of the scalar function this node calls, or `None` if it is a leaf.
    pub fn id(&self) -> Option<ScalarFnId> {
        match self {
            Self::Scalar { id, .. } => Some(*id),
            Self::Root | Self::Variable(_) => None,
        }
    }

    /// The options of the scalar function this node calls, or `None` if it is a leaf.
    pub fn options(&self) -> Option<&ErasedOptions> {
        match self {
            Self::Scalar { options, .. } => Some(options),
            Self::Root | Self::Variable(_) => None,
        }
    }

    /// Whether this node calls the scalar function with vtable `V`: its id is `V`'s and its
    /// options are of `V`'s options type.
    pub fn is<V: ScalarFnVTable + Default>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    /// The typed options for this expression if it calls the scalar function with vtable `V`.
    pub fn as_opt<V: ScalarFnVTable + Default>(&self) -> Option<&V::Options> {
        let Self::Scalar { id, options, .. } = self else {
            return None;
        };
        if *id != ScalarFnVTable::id(&V::default()) {
            return None;
        }
        options.downcast_ref::<V::Options>()
    }

    /// The typed options for this expression.
    ///
    /// # Panics
    ///
    /// Panics if the vtable type does not match.
    pub fn as_<V: ScalarFnVTable + Default>(&self) -> &V::Options {
        self.as_opt::<V>()
            .vortex_expect("Expression options type mismatch")
    }

    /// Returns the children of this expression.
    pub fn children(&self) -> &[Expression] {
        match self {
            Self::Scalar { children, .. } => children.as_slice(),
            Self::Root | Self::Variable(_) => NO_CHILDREN,
        }
    }

    /// Returns the n'th child of this expression.
    pub fn child(&self, n: usize) -> &Expression {
        &self.children()[n]
    }

    /// Replace the children of this expression with the provided new children.
    ///
    /// The number of children is checked against the function's arity when the expression is
    /// bound.
    pub fn with_children(
        self,
        children: impl IntoIterator<Item = Expression>,
    ) -> VortexResult<Self> {
        let children = Vec::from_iter(children);
        match &self {
            Self::Root => {
                vortex_ensure!(
                    children.is_empty(),
                    "Expression arity mismatch: root expects 0 children but got {}",
                    children.len()
                );
                Ok(Self::Root)
            }
            Self::Variable(name) => {
                vortex_ensure!(
                    children.is_empty(),
                    "Expression arity mismatch: variable expects 0 children but got {}",
                    children.len()
                );
                Ok(Self::Variable(name.clone()))
            }
            Self::Scalar { id, options, .. } => Ok(Self::new(*id, options.clone(), children)),
        }
    }

    /// Format the expression as a compact string.
    ///
    /// An unbound expression does not know its functions, so every node is formatted as a call
    /// of its function's id: `id(child, ..., opts=options)`. Binding the expression gives each
    /// function its own formatting.
    ///
    /// Since this is a recursive formatter, it is exposed on the public Expression type.
    pub fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => write!(f, "$"),
            Self::Variable(name) => write!(f, "@{name}"),
            Self::Scalar { id, options, .. } => fmt_call(*id, options, self, f),
        }
    }

    /// Display the expression as a formatted tree structure.
    ///
    /// This provides a hierarchical view of the expression that shows the relationships
    /// between parent and child expressions, making complex nested expressions easier
    /// to understand and debug. Children of an unbound expression are numbered, since their
    /// names come from the function; bind the expression to see them.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use vortex_array::dtype::{DType, Nullability, PType};
    /// # use vortex_array::expr::{eq, get_item, lit, not, root};
    /// let expr = not(eq(get_item("status", root()), lit("inactive")));
    ///
    /// println!("{}", expr.display_tree());
    /// ```
    ///
    /// This produces output like:
    ///
    /// ```text
    /// vortex.not()
    /// └── arg0: vortex.binary(=)
    ///     ├── arg0: vortex.get_item(status)
    ///     │   └── arg0: vortex.root()
    ///     └── arg1: vortex.literal("inactive")
    /// ```
    pub fn display_tree(&self) -> impl Display {
        DisplayTreeExpr(self)
    }

    /// Returns true if this expression contains expression E inside.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use vortex_array::scalar_fn::fns::literal::Literal;
    /// # use vortex_array::expr::{eq, lit, root};
    /// let expression = &eq(root(), lit(3u64));
    /// assert!(expression.contains::<Literal>().unwrap());
    /// let expression = root();
    /// assert!(!expression.contains::<Literal>().unwrap());
    /// ```
    pub fn contains<E: ScalarFnVTable + Default>(&self) -> VortexResult<bool> {
        let mut contains = false;
        pre_order_visit_down(self, |node| {
            if node.is::<E>() {
                contains = true;
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })?;
        Ok(contains)
    }
}

/// The error for a variable that no enclosing lambda declares.
pub(crate) fn unbound_variable(name: &FieldName) -> VortexError {
    vortex_err!("variable @{name} is not a parameter of an enclosing lambda")
}

/// The default display implementation for expressions uses the 'SQL'-style format.
impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_sql(f)
    }
}

// Switch to iterative cleanup after this many recursive drops.
const MAX_DROP_DEPTH: u32 = 32;

thread_local! {
    static DROP_DEPTH: Cell<u32> = const { Cell::new(0) };
}

// Increments the thread-local drop depth until the guard is dropped.
struct DropDepthGuard;

impl DropDepthGuard {
    fn enter() -> Option<Self> {
        DROP_DEPTH.with(|depth| {
            let current = depth.get();
            (current < MAX_DROP_DEPTH).then(|| {
                depth.set(current + 1);
                Self
            })
        })
    }
}

impl Drop for DropDepthGuard {
    fn drop(&mut self) {
        DROP_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

impl Drop for Expression {
    fn drop(&mut self) {
        let Self::Scalar { children, .. } = self else {
            return;
        };
        let Some(children) = Arc::get_mut(children) else {
            return;
        };
        if children.is_empty() {
            return;
        }

        let mut children_to_drop = std::mem::take(children);

        match DropDepthGuard::enter() {
            Some(_guard) => drop(children_to_drop),
            None => {
                while let Some(mut child) = children_to_drop.pop() {
                    if let Self::Scalar { children, .. } = &mut child
                        && let Some(expr_children) = Arc::get_mut(children)
                    {
                        children_to_drop.append(expr_children);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::cast;
    use crate::expr::lit;
    use crate::expr::not;
    use crate::expr::root;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::fns::cast::Cast;
    use crate::scalar_fn::fns::not::Not;

    fn deep_expression(depth: usize) -> Expression {
        let mut expr = lit(true);
        for _ in 0..depth {
            expr = not(expr);
        }
        expr
    }

    #[test]
    fn deep_expression_drops_within_a_small_stack() -> VortexResult<()> {
        const DEPTH: usize = 100_000;
        const STACK_SIZE: usize = 256 * 1024;

        let dropper = thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| drop(deep_expression(DEPTH)))?;

        assert!(
            dropper.join().is_ok(),
            "dropping a tree of depth {DEPTH} exhausted a {STACK_SIZE} byte stack"
        );

        Ok(())
    }

    #[test]
    fn shallow_expression_keeps_shared_children() {
        let expr = not(lit(true));
        let shared = expr.clone();

        drop(expr);

        assert_eq!(shared.children(), &[lit(true)]);
    }

    #[test]
    fn a_node_built_from_an_id_matches_the_constructor() -> VortexResult<()> {
        static CAST: CachedId = CachedId::new("vortex.cast");
        let target = DType::Primitive(PType::I64, Nullability::Nullable);

        let expr = Expression::new(*CAST, ErasedOptions::new(target.clone()), [root()]);
        assert_eq!(expr, cast(root(), target.clone()));
        assert_eq!(expr.id(), Some(*CAST));
        assert!(expr.is::<Cast>());
        assert_eq!(expr.as_opt::<Cast>(), Some(&target));
        assert!(!expr.is::<Not>());

        let scope = DType::Primitive(PType::I32, Nullability::Nullable);
        let bound = expr.bind(&scope, &VortexSession::empty())?;
        assert_eq!(bound.dtype(), &target);
        assert_eq!(
            bound,
            cast(root(), target).bind(&scope, &VortexSession::empty())?
        );
        Ok(())
    }

    #[test]
    fn options_of_the_wrong_type_are_not_the_function() {
        static CAST: CachedId = CachedId::new("vortex.cast");
        let expr = Expression::new(*CAST, ErasedOptions::new(EmptyOptions), [root()]);
        assert!(!expr.is::<Cast>());
        assert_eq!(expr.id(), Some(*CAST));

        let err = expr
            .bind(
                &DType::Bool(Nullability::NonNullable),
                &VortexSession::empty(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("takes options of type"), "{err}");
    }

    #[test]
    fn binding_checks_the_function_and_arity() {
        static MISSING: CachedId = CachedId::new("vortex.test.no_such_function");
        let scope = DType::Bool(Nullability::NonNullable);

        let err = Expression::new(*MISSING, ErasedOptions::new(EmptyOptions), [root()])
            .bind(&scope, &VortexSession::empty())
            .unwrap_err();
        assert!(err.to_string().contains("unknown scalar function"), "{err}");

        let err = not(root())
            .with_children([root(), root()])
            .unwrap()
            .bind(&scope, &VortexSession::empty())
            .unwrap_err();
        assert!(err.to_string().contains("arity mismatch"), "{err}");
    }
}
