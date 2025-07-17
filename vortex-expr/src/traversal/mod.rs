// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod references;
mod visitor;

use itertools::Itertools;
pub use references::ReferenceCollector;
pub use visitor::{pre_order_visit_down, pre_order_visit_up};
use vortex_error::VortexResult;

use crate::ExprRef;

/// Define a data fusion inspired traversal pattern for visiting nodes in a `Node`,
/// for now only VortexExpr.
///
/// This traversal is a pre-order traversal.
/// There are control traversal controls `TraversalOrder`:
/// - `Skip`: Skip visiting the children of the current node.
/// - `Stop`: Stop visiting any more nodes in the traversal.
/// - `Continue`: Continue with the traversal as expected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraversalOrder {
    /// In a top-down traversal, skip visiting the children of the current node.
    /// In the bottom-up phase of the traversal this does nothing (for now).
    Skip,
    /// Stop visiting any more nodes in the traversal.
    Stop,
    /// Continue with the traversal as expected.
    Continue,
}

impl TraversalOrder {
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
    pub value: T,
    pub order: TraversalOrder,
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
}

pub trait FolderAccumulator<T> {
    fn push(&mut self, node: T);

    fn merge_into(self, node: T) -> Self;
}

pub trait Folder<'a> {
    type NodeTy: Node;
    type O;
    type Context: FolderAccumulator<Self::O> + Clone;

    fn visit_down(
        &mut self,
        node: &'a Self::NodeTy,
        context: Self::Context,
    ) -> VortexResult<FoldDown<Self::O, Self::Context>> {
        _ = node;
        Ok(FoldDown::Continue(context))
    }

    fn visit_up(
        &mut self,
        node: &'a Self::NodeTy,
        context: Self::Context,
        children: Vec<Self::O>,
    ) -> VortexResult<FoldUp<Self::O>>;
}

pub trait FolderMut {
    type NodeTy: Node;
    type Out;
    type Context: Clone;

    fn visit_down(
        &mut self,
        _node: &Self::NodeTy,
        context: Self::Context,
    ) -> VortexResult<FoldDown<Self::Out, Self::Context>> {
        Ok(FoldDown::Continue(context))
    }

    fn visit_up(
        &mut self,
        node: Self::NodeTy,
        context: Self::Context,
        children: Vec<Self::Out>,
    ) -> VortexResult<FoldUp<Self::Out>>;
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

pub enum FoldDown<O, C> {
    /// Abort the entire traversal and immediately return the result.
    Abort(O),
    /// Continue visiting the `fold_down` of the children of the current node.
    Continue(C),
    /// Skip visiting children of the current node and return the result to the parent's `fold_up`.
    SkipChildren(O),
}

#[derive(Debug)]
pub enum FoldUp<O> {
    /// Abort the entire traversal and immediately return the result.
    Abort(O),
    /// Continue visiting the `fold_up` of the parent node.
    Continue(O),
}

impl<O> FoldUp<O> {
    pub fn into_inner(self) -> O {
        match self {
            FoldUp::Abort(out) => out,
            FoldUp::Continue(out) => out,
        }
    }
    pub fn value(self) -> O {
        self.into_inner()
    }
}

pub enum FoldState {
    Abort,
    Continue,
}

pub trait Node: Sized + Clone {
    fn apply_children<'a, F: FnMut(&'a Self) -> VortexResult<TraversalOrder>>(
        &'a self,
        f: F,
    ) -> VortexResult<TraversalOrder>;

    fn map_children<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Self>>;

    fn fold_children<B: FolderAccumulator<Self>, F: FnMut(B, &Self) -> VortexResult<FoldUp<B>>>(
        self,
        init: B,
        f: F,
    ) -> VortexResult<FoldUp<Self>>;

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
        let order = match visitor.visit_down(self)? {
            TraversalOrder::Stop => Ok(TraversalOrder::Stop),
            TraversalOrder::Skip => Ok(TraversalOrder::Continue),
            TraversalOrder::Continue => self.apply_children(|c| c.accept(visitor)),
        }?;

        match order {
            TraversalOrder::Stop | TraversalOrder::Skip => Ok(order),
            TraversalOrder::Continue => visitor.visit_up(self),
        }
    }

    fn accept_with_context<'a, V: Folder<'a, NodeTy = Self>>(
        &'a self,
        visitor: &mut V,
        context: V::Context,
    ) -> VortexResult<FoldUp<V::O>> {
        todo!()
        // match visitor.visit_down(self, context.clone())? {
        //     FoldDown::Abort(out) => return Ok(FoldUp::Abort(out)),
        //     FoldDown::SkipChildren(out) => return Ok(FoldUp::Continue(out)),
        //     FoldDown::Continue(child_context) => {
        //         let folded = self.fold_children(Vec::<V::O>::default(), |mut acc, child| {
        //             match child.accept_with_context(visitor, child_context.clone())? {
        //                 FoldUp::Abort(val) => Ok(FoldUp::Abort(vec![val])),
        //                 FoldUp::Continue(val) => {
        //                     acc.push(val);
        //                     Ok(FoldUp::Continue(acc))
        //                 }
        //             }
        //         })?;

        //         match folded {
        //             FoldUp::Abort(mut out) => return Ok(FoldUp::Abort(out.remove(0))),
        //             FoldUp::Continue(children) => visitor.visit_up(self, context, children),
        //         }
        //     }
        // }
    }

    /// A pre-order transformation
    fn transform_down<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        mut f: F,
    ) -> VortexResult<Transformed<Self>> {
        fn transform_node<N: Node, F: FnMut(N) -> VortexResult<Transformed<N>>>(
            node: N,
            f: &mut F,
        ) -> VortexResult<Transformed<N>> {
            let mut transformed = f(node)?;

            match transformed.order {
                TraversalOrder::Continue => transformed
                    .value
                    .map_children(|c| transform_node(c, f))
                    .map(|mut t| {
                        t.changed |= transformed.changed;
                        t
                    }),
                TraversalOrder::Skip => {
                    transformed.order = TraversalOrder::Continue;
                    Ok(transformed)
                }
                TraversalOrder::Stop => Ok(transformed),
            }
        }

        transform_node(self, &mut f)
    }

    /// A post-order transform
    fn transform_up<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        mut f: F,
    ) -> VortexResult<Transformed<Self>> {
        fn transform_node<N: Node, F: FnMut(N) -> VortexResult<Transformed<N>>>(
            node: N,
            f: &mut F,
        ) -> VortexResult<Transformed<N>> {
            let transformed = node.map_children(|c| transform_node(c, f))?;
            match transformed.order {
                TraversalOrder::Continue => f(transformed.value).map(|mut t| {
                    t.changed |= transformed.changed;
                    t
                }),
                TraversalOrder::Skip => Ok(transformed),
                TraversalOrder::Stop => Ok(transformed),
            }
        }

        transform_node(self, &mut f)
    }

    /// Pre-order transformation with additional state
    fn transform_with_context<V: FolderMut<NodeTy = Self>>(
        self,
        visitor: &mut V,
        context: V::Context,
    ) -> VortexResult<FoldUp<V::Out>>;
}

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

    fn fold_elements<B, F>(&'a self, init: B, f: F) -> VortexResult<FoldUp<B>>
    where
        F: FnMut(B, &'a T) -> VortexResult<FoldUp<B>>;
}

pub trait NodeRefContainer<'a, T: 'a>: Sized {
    fn apply_ref_elements<F: FnMut(&'a T) -> VortexResult<TraversalOrder>>(
        &self,
        f: F,
    ) -> VortexResult<TraversalOrder>;
}

impl<'a, T: 'a, C: NodeContainer<'a, T>> NodeRefContainer<'a, T> for Vec<&'a C> {
    fn apply_ref_elements<F: FnMut(&'a T) -> VortexResult<TraversalOrder>>(
        &self,
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

    fn fold_elements<B, F>(&'a self, init: B, mut f: F) -> VortexResult<FoldUp<B>>
    where
        F: FnMut(B, &'a T) -> VortexResult<FoldUp<B>>,
    {
        todo!()
    }
}

impl<'a> NodeContainer<'a, Self> for ExprRef {
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

    fn fold_elements<B, F: FnMut(B, &'a Self) -> VortexResult<FoldUp<B>>>(
        &'a self,
        init: B,
        mut f: F,
    ) -> VortexResult<FoldUp<B>> {
        f(init, self)
    }
}

impl Node for ExprRef {
    fn transform_with_context<V: FolderMut<NodeTy = Self>>(
        self,
        visitor: &mut V,
        context: V::Context,
    ) -> VortexResult<FoldUp<V::Out>> {
        let children = match visitor.visit_down(&self, context.clone())? {
            FoldDown::Abort(out) => return Ok(FoldUp::Abort(out)),
            FoldDown::SkipChildren(out) => return Ok(FoldUp::Continue(out)),
            FoldDown::Continue(child_context) => {
                let mut new_children = Vec::with_capacity(self.children().len());
                for child in self.children() {
                    match child
                        .clone()
                        .transform_with_context(visitor, child_context.clone())?
                    {
                        FoldUp::Abort(out) => return Ok(FoldUp::Abort(out)),
                        FoldUp::Continue(out) => new_children.push(out),
                    }
                }
                new_children
            }
        };

        visitor.visit_up(self, context, children)
    }

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
        let transformed = self
            .children()
            .into_iter()
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

    fn fold_children<B: FolderAccumulator<Self>, F: FnMut(B, &Self) -> VortexResult<FoldUp<B>>>(
        self,
        init: B,
        mut f: F,
    ) -> VortexResult<FoldUp<Self>> {
        // self.map_children(|c| )

        todo!()
    }

    // fn fold_children<'a, B, F: FnMut(B, &'a Self) -> VortexResult<FoldUp<B>>>(
    //     &'a self,
    //     mut init: B,
    //     f: F,
    // ) -> VortexResult<FoldUp<B>> {
    //     self.apply_children(|c| {

    //         match f(init, c)? {
    //             FoldUp::Abort(val) => {
    //                 init = val;
    //                 Ok(TraversalOrder::Stop)
    //             },
    //             FoldUp::Continue(val) => {
    //                 init
    //             },
    //         }
    //     })
    //     todo!()
    // }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_error::VortexResult;
    use vortex_utils::aliases::hash_set::HashSet;

    use crate::traversal::visitor::pre_order_visit_down;
    use crate::traversal::{Node, NodeRewriter, NodeVisitor, Transformed, TraversalOrder};
    use crate::{
        BinaryExpr, BinaryVTable, ExprRef, GetItemVTable, IntoExpr, LiteralExpr, LiteralVTable,
        Operator, VortexExpr, col, get_item_scope, is_root, root,
    };

    #[derive(Default)]
    pub struct ExprLitCollector<'a>(pub Vec<&'a ExprRef>);

    impl<'a> NodeVisitor<'a> for ExprLitCollector<'a> {
        type NodeTy = ExprRef;

        fn visit_down(&mut self, node: &'a ExprRef) -> VortexResult<TraversalOrder> {
            if node.is::<LiteralVTable>() {
                self.0.push(node)
            }
            Ok(TraversalOrder::Continue)
        }

        fn visit_up(&mut self, _node: &'a ExprRef) -> VortexResult<TraversalOrder> {
            Ok(TraversalOrder::Continue)
        }
    }

    fn expr_col_to_lit_transform(
        node: ExprRef,
        idx: &mut i32,
    ) -> VortexResult<Transformed<ExprRef>> {
        if node.is::<GetItemVTable>() {
            let lit_id = *idx;
            *idx += 1;
            Ok(Transformed::yes(LiteralExpr::new_expr(lit_id)))
        } else {
            Ok(Transformed::no(node))
        }
    }

    #[derive(Default)]
    pub struct SkipDownRewriter;

    impl NodeRewriter for SkipDownRewriter {
        type NodeTy = ExprRef;

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
        let col1: Arc<dyn VortexExpr> = col("col1");
        let lit1 = LiteralExpr::new(1).into_expr();
        let expr = BinaryExpr::new(col1.clone(), Operator::Eq, lit1.clone()).into_expr();
        let lit2 = LiteralExpr::new(2).into_expr();
        let expr = BinaryExpr::new(expr, Operator::And, lit2).into_expr();
        let mut printer = ExprLitCollector::default();
        expr.accept(&mut printer).unwrap();
        assert_eq!(printer.0.len(), 2);
    }

    #[test]
    fn expr_deep_mut_visitor_test() {
        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");
        let expr = BinaryExpr::new_expr(col1.clone(), Operator::Eq, col2.clone());
        let lit2 = LiteralExpr::new_expr(2);
        let expr = BinaryExpr::new_expr(expr, Operator::And, lit2);

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
        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");
        let expr1 = BinaryExpr::new_expr(col1.clone(), Operator::Eq, col2.clone());
        let col3: Arc<dyn VortexExpr> = col("col3");
        let col4: Arc<dyn VortexExpr> = col("col4");
        let expr2 = BinaryExpr::new_expr(col3.clone(), Operator::NotEq, col4.clone());
        let expr = BinaryExpr::new_expr(expr1, Operator::And, expr2);

        let mut nodes = Vec::new();
        expr.accept(&mut pre_order_visit_down(|node: &ExprRef| {
            if node.is::<GetItemVTable>() {
                nodes.push(node)
            }
            if let Some(bin) = node.as_opt::<BinaryVTable>() {
                if bin.op() == Operator::Eq {
                    return Ok(TraversalOrder::Skip);
                }
            }
            Ok(TraversalOrder::Continue)
        }))
        .unwrap();

        let nodes: HashSet<ExprRef> = HashSet::from_iter(nodes.into_iter().cloned());
        assert_eq!(
            nodes,
            HashSet::from_iter([get_item_scope("col3"), get_item_scope("col4")])
        );
    }

    #[test]
    fn expr_stop_test() {
        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");
        let expr1 = BinaryExpr::new_expr(col1.clone(), Operator::Eq, col2.clone());
        let col3: Arc<dyn VortexExpr> = col("col3");
        let col4: Arc<dyn VortexExpr> = col("col4");
        let expr2 = BinaryExpr::new_expr(col3.clone(), Operator::NotEq, col4.clone());
        let expr = BinaryExpr::new_expr(expr1, Operator::And, expr2);

        let mut nodes = Vec::new();
        expr.accept(&mut pre_order_visit_down(|node: &ExprRef| {
            if node.is::<GetItemVTable>() {
                nodes.push(node)
            }
            if let Some(bin) = node.as_opt::<BinaryVTable>() {
                if bin.op() == Operator::Eq {
                    return Ok(TraversalOrder::Stop);
                }
            }
            Ok(TraversalOrder::Continue)
        }))
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
