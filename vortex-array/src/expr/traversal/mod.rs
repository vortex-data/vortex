// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Datafusion inspired tree traversal logic.
//!
//! Users should want to implement [`Node`] and potentially [`NodeContainer`].

mod fold;
mod visitor;

use std::marker::PhantomData;

pub use fold::FoldDownContext;
pub use fold::FoldUp;
pub use fold::NodeFolderContext;
use smallvec::SmallVec;
use smallvec::smallvec;
pub use visitor::pre_order_visit_down;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::expr::BoundExpression;
use crate::expr::Expression;

/// Signal to control a traversal's flow
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalOrder {
    /// In a top-down traversal, skip visiting the children of the current node.
    /// In the bottom-up phase of the traversal, skip the next step. Either skipping the children of the node,
    /// moving to its next sibling, or skipping its parent once the children are traversed.
    Skip,
    /// Stop visiting any more nodes in the traversal.
    Stop,
    /// Continue with the traversal as expected.
    Continue,
}

impl TraversalOrder {
    /// If directed to, continue to visit nodes by running `f`, which should apply on the node's children.
    pub fn visit_children<F: FnOnce() -> VortexResult<TraversalOrder>>(
        self,
        f: F,
    ) -> VortexResult<TraversalOrder> {
        match self {
            Self::Skip => Ok(TraversalOrder::Continue),
            Self::Stop => Ok(self),
            Self::Continue => f(),
        }
    }

    /// If directed to, continue to visit nodes by running `f`, which should apply on the node's parent.
    pub fn visit_parent<F: FnOnce() -> VortexResult<TraversalOrder>>(
        self,
        f: F,
    ) -> VortexResult<TraversalOrder> {
        match self {
            Self::Continue => f(),
            Self::Skip | Self::Stop => Ok(self),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Transformed<T> {
    /// Value that was being rewritten.
    pub value: T,
    /// Controls the flow of rewriting, see [`TraversalOrder`] for more details.
    pub order: TraversalOrder,
    /// Was the value changed during rewriting.
    pub changed: bool,
}

impl<T> Transformed<T> {
    pub fn yes(value: T) -> Self {
        Self {
            value,
            order: TraversalOrder::Continue,
            changed: true,
        }
    }

    pub fn no(value: T) -> Self {
        Self {
            value,
            order: TraversalOrder::Continue,
            changed: false,
        }
    }

    pub fn into_inner(self) -> T {
        self.value
    }

    /// Apply a function to `value`, changing it without changing the `changed` field.
    pub fn map<O, F: FnOnce(T) -> O>(self, f: F) -> Transformed<O> {
        Transformed {
            value: f(self.value),
            order: self.order,
            changed: self.changed,
        }
    }
}

pub trait NodeVisitor<'a> {
    type NodeTy: Node;

    fn visit_down(&mut self, node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        _ = node;
        Ok(TraversalOrder::Continue)
    }

    fn visit_up(&mut self, node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        _ = node;
        Ok(TraversalOrder::Continue)
    }
}

pub trait NodeRewriter: Sized {
    type NodeTy: Node;

    fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        Ok(Transformed::no(node))
    }

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        Ok(Transformed::no(node))
    }
}

/// A tree node that the traversals in this module walk.
///
/// Every traversal here keeps its own stack of frames instead of recursing, so a tree deeper than
/// the thread's stack allows, such as a chain of thousands of nested expressions, still walks.
pub trait Node: Sized + Clone {
    /// The node's children, in slot order.
    fn node_children(&self) -> &[Self];

    /// Rebuild this node with new children, one per slot.
    fn with_node_children(self, children: Vec<Self>) -> VortexResult<Self>;
}

/// A node whose children are being rewritten.
struct RewriteFrame<T> {
    node: T,
    changed: bool,
    next_child: usize,
    /// The rewritten children so far. Stays `None` until a child actually changes, so untouched
    /// subtrees are never rebuilt.
    rewritten: Option<Vec<T>>,
    /// The order the last rewritten child returned.
    order: TraversalOrder,
}

impl<T: Node> RewriteFrame<T> {
    fn new(down: Transformed<T>) -> Self {
        Self {
            node: down.value,
            changed: down.changed,
            next_child: 0,
            rewritten: None,
            order: TraversalOrder::Continue,
        }
    }

    /// The next child to rewrite, if any remain.
    fn next_child(&mut self) -> Option<T> {
        let child = self.node.node_children().get(self.next_child)?.clone();
        self.next_child += 1;
        Some(child)
    }

    /// Take the result of the child that `next_child` handed out last.
    fn absorb(&mut self, child: Transformed<T>) {
        let index = self.next_child - 1;
        self.changed |= child.changed;
        self.order = child.order;
        if child.changed && self.rewritten.is_none() {
            self.rewritten = Some(self.node.node_children()[..index].to_vec());
        }
        if let Some(rewritten) = &mut self.rewritten {
            rewritten.push(child.value);
        }
        if self.order == TraversalOrder::Stop {
            // The remaining siblings are left as they are.
            if let Some(rewritten) = &mut self.rewritten {
                rewritten.extend_from_slice(&self.node.node_children()[self.next_child..]);
            }
            self.next_child = self.node.node_children().len();
        }
    }

    fn finish(self) -> VortexResult<Transformed<T>> {
        Ok(match self.rewritten {
            Some(children) => Transformed {
                value: self.node.with_node_children(children)?,
                order: self.order,
                changed: true,
            },
            None => Transformed {
                value: self.node,
                order: self.order,
                changed: self.changed,
            },
        })
    }
}

/// The frames of a bottom-up fold: a node, the next child to visit, and the children's results.
type FoldUpStack<'a, T, R> = SmallVec<[(&'a T, usize, SmallVec<[R; 4]>); 8]>;

/// A node whose children are being folded.
struct FoldFrame<T, C, R> {
    node: T,
    ctx: C,
    next_child: usize,
    results: Vec<R>,
}

pub trait NodeExt: Node {
    /// Walk the tree in pre-order (top-down) way, rewriting it as it goes.
    ///
    /// `visit_down` sees each node before its children and `visit_up` after them. A `Skip` from
    /// `visit_down` leaves the children alone but still visits the node on the way up. A `Stop`
    /// ends the walk: the remaining nodes are left as they are and no ancestor is visited on the
    /// way up.
    fn rewrite<R: NodeRewriter<NodeTy = Self>>(
        self,
        rewriter: &mut R,
    ) -> VortexResult<Transformed<Self>> {
        let mut stack: Vec<RewriteFrame<Self>> = Vec::new();
        let mut pending = Some(self);
        // The result of the subtree finished last, waiting for its parent frame.
        let mut finished: Option<Transformed<Self>> = None;

        loop {
            if let Some(node) = pending.take() {
                let down = rewriter.visit_down(node)?;
                match down.order {
                    TraversalOrder::Stop => finished = Some(down),
                    TraversalOrder::Skip => {
                        let mut up = rewriter.visit_up(down.value)?;
                        up.changed |= down.changed;
                        finished = Some(up);
                    }
                    TraversalOrder::Continue => stack.push(RewriteFrame::new(down)),
                }
            }

            if let Some(result) = finished.take() {
                let Some(frame) = stack.last_mut() else {
                    return Ok(result);
                };
                frame.absorb(result);
            }

            let frame = stack
                .last_mut()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            if let Some(child) = frame.next_child() {
                pending = Some(child);
                continue;
            }

            let frame = stack
                .pop()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            let transformed = frame.finish()?;
            finished = Some(match transformed.order {
                TraversalOrder::Stop | TraversalOrder::Skip => transformed,
                TraversalOrder::Continue => {
                    let mut up = rewriter.visit_up(transformed.value)?;
                    up.changed |= transformed.changed;
                    up
                }
            });
        }
    }

    /// A pre-order (top-down) traversal.
    ///
    /// `visit_down` sees each node before its children and `visit_up` after them. A `Skip` from
    /// `visit_down` leaves the children alone but still visits the node on the way up. A `Stop`
    /// from either ends the walk without visiting any ancestor on the way up.
    fn accept<'a, V: NodeVisitor<'a, NodeTy = Self>>(
        &'a self,
        visitor: &mut V,
    ) -> VortexResult<TraversalOrder> {
        let mut stack: Vec<(&'a Self, usize)> = Vec::new();
        let mut pending = Some(self);
        let mut finished: Option<TraversalOrder> = None;

        loop {
            if let Some(node) = pending.take() {
                match visitor.visit_down(node)? {
                    TraversalOrder::Stop => finished = Some(TraversalOrder::Stop),
                    TraversalOrder::Skip => finished = Some(visitor.visit_up(node)?),
                    TraversalOrder::Continue => stack.push((node, 0)),
                }
            }

            if let Some(order) = finished.take() {
                if stack.is_empty() {
                    return Ok(order);
                }
                if order == TraversalOrder::Stop {
                    stack.pop();
                    finished = Some(TraversalOrder::Stop);
                    continue;
                }
            }

            let (node, next_child) = stack
                .last_mut()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            let node: &'a Self = node;
            if let Some(child) = node.node_children().get(*next_child) {
                *next_child += 1;
                pending = Some(child);
                continue;
            }

            stack.pop();
            finished = Some(visitor.visit_up(node)?);
        }
    }

    /// A pre-order transformation
    fn transform_down<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Self>> {
        let mut rewriter = FnRewriter {
            f,
            _data: PhantomData,
        };

        self.rewrite(&mut rewriter)
    }

    /// applies the `NodeFolderContext` to the Node tree, with an initial `Context`.
    ///
    /// A `Stop` from any node ends the fold with that result, without visiting any ancestor on
    /// the way up.
    fn fold_context<R, F: NodeFolderContext<NodeTy = Self, Result = R>>(
        self,
        ctx: &F::Context,
        folder: &mut F,
    ) -> VortexResult<FoldUp<R>> {
        let mut stack: Vec<FoldFrame<Self, F::Context, R>> = Vec::new();
        let mut pending = Some(self);
        let mut finished: Option<FoldUp<R>> = None;

        loop {
            if let Some(node) = pending.take() {
                let parent_ctx = stack.last().map_or(ctx, |frame| &frame.ctx);
                match folder.visit_down(parent_ctx, &node)? {
                    FoldDownContext::Continue(child_ctx) => {
                        let results = Vec::with_capacity(node.node_children().len());
                        stack.push(FoldFrame {
                            node,
                            ctx: child_ctx,
                            next_child: 0,
                            results,
                        });
                    }
                    FoldDownContext::Skip(result) => finished = Some(FoldUp::Continue(result)),
                    FoldDownContext::Stop(result) => finished = Some(FoldUp::Stop(result)),
                }
            }

            if let Some(result) = finished.take() {
                let Some(frame) = stack.last_mut() else {
                    return Ok(result);
                };
                match result {
                    FoldUp::Stop(result) => return Ok(FoldUp::Stop(result)),
                    FoldUp::Continue(result) => frame.results.push(result),
                }
            }

            let frame = stack
                .last_mut()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            if let Some(child) = frame.node.node_children().get(frame.next_child) {
                frame.next_child += 1;
                pending = Some(child.clone());
                continue;
            }

            let frame = stack
                .pop()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            finished = Some(folder.visit_up(frame.node, &frame.ctx, frame.results)?);
        }
    }

    /// Fold the tree bottom-up: `f` sees each node with the results of its children in slot
    /// order, and the root's result is returned.
    fn fold_up<'a, R>(
        &'a self,
        mut f: impl FnMut(&'a Self, SmallVec<[R; 4]>) -> VortexResult<R>,
    ) -> VortexResult<R> {
        let mut stack: FoldUpStack<'a, Self, R> =
            smallvec![(self, 0, SmallVec::with_capacity(self.node_children().len()))];

        loop {
            let (node, next_child, _) = stack
                .last_mut()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            let node: &'a Self = node;
            if let Some(child) = node.node_children().get(*next_child) {
                *next_child += 1;
                stack.push((
                    child,
                    0,
                    SmallVec::with_capacity(child.node_children().len()),
                ));
                continue;
            }

            let (node, _, results) = stack
                .pop()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            let result = f(node, results)?;
            match stack.last_mut() {
                Some((_, _, parent_results)) => parent_results.push(result),
                None => return Ok(result),
            }
        }
    }
}

impl<T: Node> NodeExt for T {}

struct FnRewriter<F, T> {
    f: F,
    _data: PhantomData<T>,
}

impl<F, T> NodeRewriter for FnRewriter<F, T>
where
    T: Node,
    F: FnMut(T) -> VortexResult<Transformed<T>>,
{
    type NodeTy = T;

    fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        (self.f)(node)
    }
}

impl Node for Expression {
    fn node_children(&self) -> &[Self] {
        Expression::children(self)
    }

    fn with_node_children(self, children: Vec<Self>) -> VortexResult<Self> {
        Expression::with_children(self, children)
    }
}

impl Node for BoundExpression {
    fn node_children(&self) -> &[Self] {
        BoundExpression::children(self)
    }

    fn with_node_children(self, children: Vec<Self>) -> VortexResult<Self> {
        BoundExpression::with_children(self, children)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::NodeExt;
    use super::NodeRewriter;
    use super::NodeVisitor;
    use super::Transformed;
    use super::TraversalOrder;
    use super::visitor::pre_order_visit_down;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::is_root;
    use crate::expr::lit;
    use crate::expr::not_eq;
    use crate::expr::root;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::get_item::GetItem;
    use crate::scalar_fn::fns::literal::Literal;
    use crate::scalar_fn::fns::operators::Operator;

    #[derive(Default)]
    pub struct ExprLitCollector<'a>(pub Vec<&'a Expression>);

    impl<'a> NodeVisitor<'a> for ExprLitCollector<'a> {
        type NodeTy = Expression;

        fn visit_down(&mut self, node: &'a Expression) -> VortexResult<TraversalOrder> {
            if node.is::<Literal>() {
                self.0.push(node)
            }
            Ok(TraversalOrder::Continue)
        }

        fn visit_up(&mut self, _node: &'a Expression) -> VortexResult<TraversalOrder> {
            Ok(TraversalOrder::Continue)
        }
    }

    fn expr_col_to_lit_transform(
        node: Expression,
        idx: &mut i32,
    ) -> VortexResult<Transformed<Expression>> {
        if node.is::<GetItem>() {
            let lit_id = *idx;
            *idx += 1;
            Ok(Transformed::yes(lit(lit_id)))
        } else {
            Ok(Transformed::no(node))
        }
    }

    #[derive(Default)]
    pub struct SkipDownRewriter;

    impl NodeRewriter for SkipDownRewriter {
        type NodeTy = Expression;

        fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
            Ok(Transformed {
                value: node,
                order: TraversalOrder::Skip,
                changed: false,
            })
        }

        fn visit_up(&mut self, _node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
            Ok(Transformed::yes(root()))
        }
    }

    #[test]
    fn expr_deep_visitor_test() {
        let col1: Expression = col("col1");
        let lit1 = lit(1);
        let expr = eq(col1, lit1);
        let lit2 = lit(2);
        let expr = and(expr, lit2);
        let mut printer = ExprLitCollector::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(printer.0, vec![&lit(1), &lit(2)]);
    }

    #[test]
    fn expr_deep_mut_visitor_test() {
        let col1: Expression = col("col1");
        let col2: Expression = col("col2");
        let expr = eq(col1, col2);
        let lit2 = lit(2);
        let expr = and(expr, lit2);

        let mut idx = 0_i32;
        let new = expr
            .transform_down(|node| expr_col_to_lit_transform(node, &mut idx))
            .unwrap();
        assert!(new.changed);

        let expr = new.value;

        let mut printer = ExprLitCollector::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(printer.0, vec![&lit(0), &lit(1), &lit(2)]);
        assert_eq!(expr, and(eq(lit(0), lit(1)), lit(2)));
    }

    #[test]
    fn expr_skip_test() {
        let col1: Expression = col("col1");
        let col2: Expression = col("col2");
        let expr1 = eq(col1, col2);
        let col3: Expression = col("col3");
        let col4: Expression = col("col4");
        let expr2 = not_eq(col3, col4);
        let expr = and(expr1, expr2);

        let mut nodes = Vec::new();
        pre_order_visit_down(&expr, |node: &Expression| {
            if node.is::<GetItem>() {
                nodes.push(node)
            }
            if let Some(operator) = node.as_opt::<Binary>()
                && *operator == Operator::Eq
            {
                return Ok(TraversalOrder::Skip);
            }
            Ok(TraversalOrder::Continue)
        })
        .unwrap();

        let nodes: HashSet<Expression> = HashSet::from_iter(nodes.into_iter().cloned());
        assert_eq!(nodes, HashSet::from_iter([col("col3"), col("col4")]));
    }

    #[test]
    fn expr_stop_test() {
        let col1: Expression = col("col1");
        let col2: Expression = col("col2");
        let expr1 = eq(col1, col2);
        let col3: Expression = col("col3");
        let col4: Expression = col("col4");
        let expr2 = not_eq(col3, col4);
        let expr = and(expr1, expr2);

        let mut nodes = Vec::new();
        pre_order_visit_down(&expr, |node: &Expression| {
            if node.is::<GetItem>() {
                nodes.push(node)
            }
            if let Some(operator) = node.as_opt::<Binary>()
                && *operator == Operator::Eq
            {
                return Ok(TraversalOrder::Stop);
            }
            Ok(TraversalOrder::Continue)
        })
        .unwrap();

        assert!(nodes.is_empty());
    }

    #[test]
    fn expr_skip_down_visit_up() {
        let col = col("col");

        let mut visitor = SkipDownRewriter;
        let result = col.rewrite(&mut visitor).unwrap();

        assert!(result.changed);
        assert!(is_root(&result.value));
    }
}
