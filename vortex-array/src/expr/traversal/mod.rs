// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Datafusion inspired tree traversal logic.
//!
//! Users should want to implement [`Node`] and potentially [`NodeContainer`].

mod fold;
mod visitor;

use std::marker::PhantomData;
use std::sync::Arc;

pub use fold::FoldDownContext;
pub use fold::FoldUp;
pub use fold::NodeFolderContext;
use smallvec::SmallVec;
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

pub trait Node: Sized + Clone {
    /// Walk the node's children by applying `f` to them.
    ///
    /// This is a lower level API that other functions rely on for their implementation.
    fn apply_children<'a, F: FnMut(&'a Self) -> VortexResult<TraversalOrder>>(
        &'a self,
        f: F,
    ) -> VortexResult<TraversalOrder>;

    /// Rewrite the node's children by applying `f` to them.
    ///
    /// This is a lower level API that other functions rely on for their implementation.
    fn map_children<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Self>>;

    /// This is a lower level API that other functions rely on for their implementation.
    fn iter_children<T>(&self, f: impl FnOnce(&mut dyn Iterator<Item = &Self>) -> T) -> T;

    /// This is a lower level API that other functions rely on for their implementation.
    fn children_count(&self) -> usize;
}

/// The children of a node, listed once per frame of an iterative walk.
type Children<T> = SmallVec<[T; 4]>;

/// A node whose children are being rewritten.
struct RewriteFrame<T> {
    /// The node after `visit_down`.
    down: Transformed<T>,
    children: Children<T>,
    next_child: usize,
    /// The rewritten children so far, in order. Stops at the child that returned `Stop`.
    rewritten: Children<Transformed<T>>,
}

impl<T: Node> RewriteFrame<T> {
    fn new(down: Transformed<T>) -> Self {
        let children = down
            .value
            .iter_children(|children| children.cloned().collect());
        Self {
            down,
            children,
            next_child: 0,
            rewritten: Children::new(),
        }
    }

    /// The next child to rewrite, if any remain.
    fn next_child(&mut self) -> Option<T> {
        let child = self.children.get(self.next_child)?.clone();
        self.next_child += 1;
        Some(child)
    }

    /// Take the result of the child that `next_child` handed out last.
    fn absorb(&mut self, child: Transformed<T>) {
        if child.order == TraversalOrder::Stop {
            // The remaining siblings are left as they are.
            self.next_child = self.children.len();
        }
        self.rewritten.push(child);
    }

    /// Rebuild the node over its rewritten children, replaying their results through
    /// [`Node::map_children`] in order.
    fn finish(self) -> VortexResult<Transformed<T>> {
        let Self {
            down, rewritten, ..
        } = self;
        let mut rewritten = rewritten.into_iter();
        let mut transformed = down.value.map_children(|_child| {
            Ok(rewritten
                .next()
                .vortex_expect("every visited child has a result"))
        })?;
        transformed.changed |= down.changed;
        Ok(transformed)
    }
}

/// A node whose children are being folded.
struct FoldFrame<T, C, R> {
    node: T,
    ctx: C,
    children: Children<T>,
    next_child: usize,
    results: Vec<R>,
}

/// The frames of a bottom-up fold: a node, its children, the next child to visit, and the
/// children's results.
type FoldUpStack<'a, T, R> = SmallVec<[(&'a T, Children<&'a T>, usize, SmallVec<[R; 4]>); 8]>;

/// The children of `node` by reference, through [`Node::apply_children`].
fn child_refs<T: Node>(node: &T) -> VortexResult<Children<&T>> {
    let mut children = Children::new();
    node.apply_children(|child| {
        children.push(child);
        Ok(TraversalOrder::Continue)
    })?;
    Ok(children)
}

pub trait NodeExt: Node {
    /// Walk the tree in pre-order (top-down) way, rewriting it as it goes.
    ///
    /// `visit_down` sees each node before its children and `visit_up` after them. A `Skip` from
    /// `visit_down` leaves the children alone but still visits the node on the way up. A `Stop`
    /// leaves the remaining nodes as they are.
    ///
    /// The walk keeps its own stack of frames instead of recursing, so a tree deeper than the
    /// thread's stack allows, such as a chain of thousands of nested expressions, still rewrites.
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
                let mut down = rewriter.visit_down(node)?;
                match down.order {
                    TraversalOrder::Stop => finished = Some(down),
                    TraversalOrder::Skip => {
                        down.order = TraversalOrder::Continue;
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
    /// ends the walk.
    ///
    /// The walk keeps its own stack of frames instead of recursing, so a tree deeper than the
    /// thread's stack allows, such as a chain of thousands of nested expressions, still walks.
    fn accept<'a, V: NodeVisitor<'a, NodeTy = Self>>(
        &'a self,
        visitor: &mut V,
    ) -> VortexResult<TraversalOrder> {
        struct Frame<'a, T> {
            node: &'a T,
            children: Children<&'a T>,
            next_child: usize,
            /// The orders the visited children returned, replayed through `apply_children`.
            orders: Children<TraversalOrder>,
        }

        let mut stack: Vec<Frame<'a, Self>> = Vec::new();
        let mut pending = Some(self);
        let mut finished: Option<TraversalOrder> = None;

        loop {
            if let Some(node) = pending.take() {
                match visitor.visit_down(node)? {
                    TraversalOrder::Stop => finished = Some(TraversalOrder::Stop),
                    TraversalOrder::Skip => finished = Some(visitor.visit_up(node)?),
                    TraversalOrder::Continue => stack.push(Frame {
                        node,
                        children: child_refs(node)?,
                        next_child: 0,
                        orders: Children::new(),
                    }),
                }
            }

            if let Some(order) = finished.take() {
                let Some(frame) = stack.last_mut() else {
                    return Ok(order);
                };
                if order == TraversalOrder::Stop {
                    frame.next_child = frame.children.len();
                }
                frame.orders.push(order);
            }

            let frame = stack
                .last_mut()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            if let Some(child) = frame.children.get(frame.next_child) {
                frame.next_child += 1;
                pending = Some(child);
                continue;
            }

            let frame = stack
                .pop()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            let mut orders = frame.orders.into_iter();
            finished = Some(
                frame
                    .node
                    .apply_children(|_child| {
                        Ok(orders
                            .next()
                            .vortex_expect("every visited child has an order"))
                    })?
                    .visit_parent(|| visitor.visit_up(frame.node))?,
            );
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
    ///
    /// The fold keeps its own stack of frames instead of recursing, so a tree deeper than the
    /// thread's stack allows, such as a chain of thousands of nested expressions, still folds.
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
                        let children: Children<Self> =
                            node.iter_children(|children| children.cloned().collect());
                        let results = Vec::with_capacity(children.len());
                        stack.push(FoldFrame {
                            node,
                            ctx: child_ctx,
                            children,
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
            if let Some(child) = frame.children.get(frame.next_child) {
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
    ///
    /// The fold keeps its own stack of frames instead of recursing, so a tree deeper than the
    /// thread's stack allows, such as a chain of thousands of nested expressions, still folds.
    fn fold_up<'a, R>(
        &'a self,
        mut f: impl FnMut(&'a Self, SmallVec<[R; 4]>) -> VortexResult<R>,
    ) -> VortexResult<R> {
        let mut stack: FoldUpStack<'a, Self, R> = SmallVec::new();
        let children = child_refs(self)?;
        stack.push((
            self,
            children,
            0,
            SmallVec::with_capacity(self.children_count()),
        ));

        loop {
            let (_, children, next_child, _) = stack
                .last_mut()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            if let Some(child) = children.get(*next_child) {
                let child: &'a Self = child;
                *next_child += 1;
                let grandchildren = child_refs(child)?;
                stack.push((
                    child,
                    grandchildren,
                    0,
                    SmallVec::with_capacity(child.children_count()),
                ));
                continue;
            }

            let (node, _, _, results) = stack
                .pop()
                .vortex_expect("a frame is on the stack until its subtree is finished");
            let result = f(node, results)?;
            match stack.last_mut() {
                Some((_, _, _, parent_results)) => parent_results.push(result),
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

/// A container holding a [`Node`]'s children, which a function can be applied (or mapped) to.
///
/// The trait is also implemented to container types in order to make implementing [`Node::map_children`]
/// and [`Node::apply_children`] easier.
pub trait NodeContainer<'a, T: 'a>: Sized {
    /// Applies `f` to all elements of the container, accepting them by reference
    fn apply_elements<F: FnMut(&'a T) -> VortexResult<TraversalOrder>>(
        &'a self,
        f: F,
    ) -> VortexResult<TraversalOrder>;

    /// Consumes all the children of the node, replacing them with the result of `f`.
    fn map_elements<F: FnMut(T) -> VortexResult<Transformed<T>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Self>>;
}

pub trait NodeRefContainer<'a, T: 'a>: Sized {
    fn apply_ref_elements<F: FnMut(&'a T) -> VortexResult<TraversalOrder>>(
        &self,
        f: F,
    ) -> VortexResult<TraversalOrder>;
}

impl<'a, T: 'a, C: NodeContainer<'a, T>> NodeRefContainer<'a, T> for &'a [C] {
    fn apply_ref_elements<F: FnMut(&'a T) -> VortexResult<TraversalOrder>>(
        &self,
        mut f: F,
    ) -> VortexResult<TraversalOrder> {
        let mut order = TraversalOrder::Continue;

        for c in *self {
            order = c.apply_elements(&mut f)?;
            match order {
                TraversalOrder::Continue | TraversalOrder::Skip => {}
                TraversalOrder::Stop => return Ok(TraversalOrder::Stop),
            }
        }

        Ok(order)
    }
}

impl<'a, T: 'a, C: NodeContainer<'a, T>> NodeContainer<'a, T> for Box<C> {
    fn apply_elements<F: FnMut(&'a T) -> VortexResult<TraversalOrder>>(
        &'a self,
        f: F,
    ) -> VortexResult<TraversalOrder> {
        self.as_ref().apply_elements(f)
    }

    fn map_elements<F: FnMut(T) -> VortexResult<Transformed<T>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Box<C>>> {
        Ok((*self).map_elements(f)?.map(Box::new))
    }
}

impl<'a, T, C> NodeContainer<'a, T> for Arc<C>
where
    T: 'a,
    C: NodeContainer<'a, T> + Clone,
{
    fn apply_elements<F: FnMut(&'a T) -> VortexResult<TraversalOrder>>(
        &'a self,
        f: F,
    ) -> VortexResult<TraversalOrder> {
        self.as_ref().apply_elements(f)
    }

    fn map_elements<F: FnMut(T) -> VortexResult<Transformed<T>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Arc<C>>> {
        Ok(Arc::unwrap_or_clone(self).map_elements(f)?.map(Arc::new))
    }
}

impl<'a, T: 'a, C: NodeContainer<'a, T>> NodeContainer<'a, T> for [C; 2] {
    fn apply_elements<F: FnMut(&'a T) -> VortexResult<TraversalOrder>>(
        &'a self,
        mut f: F,
    ) -> VortexResult<TraversalOrder> {
        let [lhs, rhs] = self;
        match lhs.apply_elements(&mut f)? {
            TraversalOrder::Skip | TraversalOrder::Continue => rhs.apply_elements(&mut f),
            TraversalOrder::Stop => Ok(TraversalOrder::Stop),
        }
    }

    fn map_elements<F: FnMut(T) -> VortexResult<Transformed<T>>>(
        self,
        mut f: F,
    ) -> VortexResult<Transformed<[C; 2]>> {
        let [lhs, rhs] = self;
        let transformed = lhs.map_elements(&mut f)?;
        match transformed.order {
            TraversalOrder::Skip | TraversalOrder::Continue => {
                let mut t = rhs.map_elements(&mut f)?;
                t.changed |= transformed.changed;
                Ok(t.map(|new_lhs| [new_lhs, transformed.value]))
            }
            TraversalOrder::Stop => Ok(transformed.map(|new_lhs| [new_lhs, rhs])),
        }
    }
}

impl<'a, T: 'a, C: NodeContainer<'a, T>> NodeContainer<'a, T> for Vec<C> {
    fn apply_elements<F: FnMut(&'a T) -> VortexResult<TraversalOrder>>(
        &'a self,
        mut f: F,
    ) -> VortexResult<TraversalOrder> {
        let mut order = TraversalOrder::Continue;

        for c in self {
            order = c.apply_elements(&mut f)?;
            match order {
                TraversalOrder::Continue | TraversalOrder::Skip => {}
                TraversalOrder::Stop => return Ok(TraversalOrder::Stop),
            }
        }

        Ok(order)
    }

    fn map_elements<F: FnMut(T) -> VortexResult<Transformed<T>>>(
        self,
        mut f: F,
    ) -> VortexResult<Transformed<Self>> {
        let mut order = TraversalOrder::Continue;
        let mut changed = false;

        let value = self
            .into_iter()
            .map(|c| match order {
                TraversalOrder::Continue | TraversalOrder::Skip => {
                    c.map_elements(&mut f).map(|result| {
                        order = result.order;
                        changed |= result.changed;
                        result.value
                    })
                }
                TraversalOrder::Stop => Ok(c),
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Transformed {
            value,
            order,
            changed,
        })
    }
}

impl<'a> NodeContainer<'a, Self> for Expression {
    fn apply_elements<F: FnMut(&'a Self) -> VortexResult<TraversalOrder>>(
        &'a self,
        mut f: F,
    ) -> VortexResult<TraversalOrder> {
        f(self)
    }

    fn map_elements<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        mut f: F,
    ) -> VortexResult<Transformed<Self>> {
        f(self)
    }
}

impl Node for Expression {
    fn apply_children<'a, F: FnMut(&'a Self) -> VortexResult<TraversalOrder>>(
        &'a self,
        mut f: F,
    ) -> VortexResult<TraversalOrder> {
        self.children().apply_ref_elements(&mut f)
    }

    fn map_children<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Self>> {
        let transformed = self.children().to_vec().map_elements(f)?;

        if transformed.changed {
            Ok(Transformed {
                value: self.with_children(transformed.value)?,
                order: transformed.order,
                changed: true,
            })
        } else {
            Ok(Transformed::no(self))
        }
    }

    fn iter_children<T>(&self, f: impl FnOnce(&mut dyn Iterator<Item = &Self>) -> T) -> T {
        f(&mut self.children().iter())
    }

    fn children_count(&self) -> usize {
        self.children().len()
    }
}

impl Node for BoundExpression {
    fn apply_children<'a, F: FnMut(&'a Self) -> VortexResult<TraversalOrder>>(
        &'a self,
        mut f: F,
    ) -> VortexResult<TraversalOrder> {
        let BoundExpression::Scalar { children, .. } = self else {
            return Ok(TraversalOrder::Continue);
        };

        for child in children.iter() {
            match f(child)? {
                TraversalOrder::Continue | TraversalOrder::Skip => {}
                TraversalOrder::Stop => return Ok(TraversalOrder::Stop),
            }
        }

        Ok(TraversalOrder::Continue)
    }

    fn map_children<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        mut f: F,
    ) -> VortexResult<Transformed<Self>> {
        let BoundExpression::Scalar { children, .. } = &self else {
            return Ok(Transformed::no(self));
        };

        let mut order = TraversalOrder::Continue;
        // Stays `None` until a child actually changes. Most nodes of a rewritten tree are
        // untouched.
        let mut rewritten: Option<Vec<Self>> = None;

        for (index, child) in children.iter().enumerate() {
            let value = match order {
                TraversalOrder::Continue | TraversalOrder::Skip => {
                    let result = f(child.clone())?;
                    order = result.order;
                    if result.changed && rewritten.is_none() {
                        let mut prefix = Vec::with_capacity(children.len());
                        prefix.extend_from_slice(&children[..index]);
                        rewritten = Some(prefix);
                    }
                    result.value
                }
                TraversalOrder::Stop => child.clone(),
            };

            if let Some(rewritten) = &mut rewritten {
                rewritten.push(value);
            }
        }

        match rewritten {
            Some(rewritten) => Ok(Transformed {
                value: self.with_children(rewritten)?,
                order,
                changed: true,
            }),
            None => Ok(Transformed::no(self)),
        }
    }

    fn iter_children<T>(&self, f: impl FnOnce(&mut dyn Iterator<Item = &Self>) -> T) -> T {
        match self {
            BoundExpression::Scalar { children, .. } => f(&mut children.iter()),
            BoundExpression::Root { .. } => f(&mut std::iter::empty()),
        }
    }

    fn children_count(&self) -> usize {
        match self {
            BoundExpression::Scalar { children, .. } => children.len(),
            BoundExpression::Root { .. } => 0,
        }
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
