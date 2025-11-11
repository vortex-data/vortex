// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Datafusion inspired tree traversal logic.
//!
//! Users should want to implement [`Node`] and potentially [`NodeContainer`].

mod fold;
mod references;
mod visitor;

use std::marker::PhantomData;
use std::sync::Arc;

pub use fold::{FoldDown, FoldDownContext, FoldUp, NodeFolder, NodeFolderContext};
use itertools::Itertools;
pub use references::ReferenceCollector;
pub use visitor::{pre_order_visit_down, pre_order_visit_up};
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::traversal::fold::NodeFolderContextWrapper;

/// Signal to control a traversal's flow
#[derive(Debug, Clone, PartialEq, Eq)]
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

pub trait NodeExt: Node {
    /// Walk the tree in pre-order (top-down) way, rewriting it as it goes.
    fn rewrite<R: NodeRewriter<NodeTy = Self>>(
        self,
        rewriter: &mut R,
    ) -> VortexResult<Transformed<Self>> {
        let mut transformed = rewriter.visit_down(self)?;

        let transformed = match transformed.order {
            TraversalOrder::Stop => Ok(transformed),
            TraversalOrder::Skip => {
                transformed.order = TraversalOrder::Continue;
                Ok(transformed)
            }
            TraversalOrder::Continue => transformed
                .value
                .map_children(|c| c.rewrite(rewriter))
                .map(|mut t| {
                    t.changed |= transformed.changed;
                    t
                }),
        }?;

        match transformed.order {
            TraversalOrder::Stop | TraversalOrder::Skip => Ok(transformed),
            TraversalOrder::Continue => {
                let mut up_rewrite = rewriter.visit_up(transformed.value)?;
                up_rewrite.changed |= transformed.changed;
                Ok(up_rewrite)
            }
        }
    }

    /// A pre-order (top-down) traversal.
    fn accept<'a, V: NodeVisitor<'a, NodeTy = Self>>(
        &'a self,
        visitor: &mut V,
    ) -> VortexResult<TraversalOrder> {
        visitor
            .visit_down(self)?
            .visit_children(|| self.apply_children(|c| c.accept(visitor)))?
            .visit_parent(|| visitor.visit_up(self))
    }

    /// A pre-order transformation
    fn transform_down<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Self>> {
        let mut rewriter = FnRewriter {
            f_down: Some(f),
            f_up: None,
            _data: PhantomData,
        };

        self.rewrite(&mut rewriter)
    }

    /// A post-order transform
    fn transform_up<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Self>> {
        let mut rewriter = FnRewriter {
            f_down: None,
            f_up: Some(f),
            _data: PhantomData,
        };

        self.rewrite(&mut rewriter)
    }

    /// applies the `NodeFolderContext` to the Node tree, with an initial `Context`.
    fn fold_context<R, F: NodeFolderContext<NodeTy = Self, Result = R>>(
        self,
        ctx: &F::Context,
        folder: &mut F,
    ) -> VortexResult<FoldUp<R>> {
        let transformed = folder.visit_down(ctx, &self)?;
        let ctx = match transformed {
            FoldDownContext::Continue(ctx) => ctx,
            FoldDownContext::Skip(r) => return Ok(FoldUp::Continue(r)),
            FoldDownContext::Stop(r) => return Ok(FoldUp::Stop(r)),
        };

        let mut children = Vec::with_capacity(self.children_count());
        let mut stop_result = None;
        self.iter_children(|children_iter| -> VortexResult<()> {
            for c in children_iter {
                let t = c.clone().fold_context(&ctx, folder)?;
                match t {
                    FoldUp::Stop(r) => {
                        stop_result = Some(r);
                        return Ok(());
                    }
                    FoldUp::Continue(r) => {
                        children.push(r);
                    }
                }
            }
            Ok(())
        })?;

        if let Some(result) = stop_result {
            return Ok(FoldUp::Stop(result));
        }

        folder.visit_up(self, &ctx, children)
    }

    /// applies the `NodeFolder` to the Node tree
    fn fold<R, F: NodeFolder<NodeTy = Self, Result = R>>(
        self,
        folder: &mut F,
    ) -> VortexResult<FoldUp<R>> {
        let mut folder = NodeFolderContextWrapper { inner: folder };
        self.fold_context(&(), &mut folder)
    }
}

impl<T: Node> NodeExt for T {}

struct FnRewriter<F, T> {
    f_down: Option<F>,
    f_up: Option<F>,
    _data: PhantomData<T>,
}

impl<F, T> NodeRewriter for FnRewriter<F, T>
where
    T: Node,
    F: FnMut(T) -> VortexResult<Transformed<T>>,
{
    type NodeTy = T;

    fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        if let Some(f) = self.f_down.as_mut() {
            f(node)
        } else {
            Ok(Transformed::no(node))
        }
    }

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        if let Some(f) = self.f_up.as_mut() {
            f(node)
        } else {
            Ok(Transformed::no(node))
        }
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
        self.children().as_ref().apply_ref_elements(&mut f)
    }

    fn map_children<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Self>> {
        let transformed = self
            .children()
            .iter()
            .cloned()
            .collect_vec()
            .map_elements(f)?;

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

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::visitor::pre_order_visit_down;
    use super::{NodeExt, NodeRewriter, NodeVisitor, Transformed, TraversalOrder};
    use crate::expr::Expression;
    use crate::expr::exprs::binary::{Binary, and, eq, not_eq};
    use crate::expr::exprs::get_item::{GetItem, col};
    use crate::expr::exprs::literal::{Literal, lit};
    use crate::expr::exprs::operators::Operator;
    use crate::expr::exprs::root::{is_root, root};

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
        assert_eq!(printer.0.len(), 2);
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
            .transform_up(|node| expr_col_to_lit_transform(node, &mut idx))
            .unwrap();
        assert!(new.changed);

        let expr = new.value;

        let mut printer = ExprLitCollector::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(printer.0.len(), 3);
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
            if let Some(bin) = node.as_opt::<Binary>()
                && bin.operator() == Operator::Eq
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
            if let Some(bin) = node.as_opt::<Binary>()
                && bin.operator() == Operator::Eq
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
