use std::iter;

use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_dtype::{Field, FieldName, FieldNames};
use vortex_error::VortexResult;

use crate::traversal::{FoldChildren, FoldDown, FoldUp, Folder};
use crate::{ExprRef, GetItem, Identity};

/// Given an expression, an identity-type and a list of n fields return n optional expressions
/// ones containing only references to the corresponding field and an expression defined in terms of
/// the n expression which combines them back into a single expression.

fn split_expression(
    _expr: ExprRef,
    _fields: &FieldNames,
) -> (ExprRef, HashMap<FieldName, ExprRef>) {
    // st {
    //   a:
    //   b: {
    //     c:
    //     d:
    //   }
    // }

    // f(id.a) /\ g(id) ==> f({a: id.a, b: id.b}.a) /\ g({a: id.a, b: id.b})

    // e_1.1 /\ g({a: e_2.1, b: e_1.2}) where, e_1 = {1: f(id), 2: id} in a and e_2 = {1: id} in b

    // x > 5 and x < 10 and y > 5
    // let e1 = x > 5 /\ x < 10 in let e2 = y > 5 in e1 /\ e2

    // x.a > 5 and y > 5 and x.b < 10
    // let e1 = pack(e11: x.a > 5, e12: x.b < 10) in let e2 = pack(e22: y > 5) in e1.e11 /\ e2.e22 /\ e1.e12
    todo!()
}

struct ExprSplitter {
    field: Field,
    sub_expressions: Vec<ExprRef>,
}

impl ExprSplitter {
    fn new(field: Field) -> Self {
        Self {
            field,
            sub_expressions: vec![],
        }
    }
}

// Hashmap from expr to [get_item(field, Identity)]

// impl Folder for ExprSplitter {
//     type NodeTy = ExprRef;
//     type Out = ExprRef;
//     type Context = Option<Field>;
//
//     fn visit_down(
//         &mut self,
//         node: &Self::NodeTy,
//         context: Self::Context,
//     ) -> VortexResult<FoldDown<Self::Out, Self::Context>> {
//         node.references().contains(&self.field)
//     }
//
//     fn visit_up(
//         &mut self,
//         node: Self::NodeTy,
//         context: Self::Context,
//         children: FoldChildren<Self::Out>,
//     ) -> VortexResult<FoldUp<Self::Out>> {
//         todo!()
//     }
// }

#[derive(Clone, Debug)]
enum FieldAccesses {
    Fields(HashSet<Field>),
    AllFields,
}

impl FieldAccesses {
    fn union(self, other: &Self) -> Self {
        match (self, other) {
            (FieldAccesses::AllFields, _) => FieldAccesses::AllFields,
            (_, FieldAccesses::AllFields) => FieldAccesses::AllFields,
            (FieldAccesses::Fields(fields1), FieldAccesses::Fields(fields2)) => {
                FieldAccesses::Fields(fields1.union(fields2).cloned().collect())
            }
        }
    }
}

// For all subexpressions in an expression find the fields access directly on identity
struct ExprTopLevelRef<'a> {
    sub_expressions: HashMap<&'a ExprRef, FieldNames>,
}

impl<'a> ExprTopLevelRef<'a> {
    fn new() -> Self {
        Self {
            sub_expressions: HashMap::new(),
        }
    }
}

impl<'a> Folder<'a> for ExprTopLevelRef<'a> {
    type NodeTy = ExprRef;
    type Out = FieldAccesses;
    type Context = Option<Field>;

    fn visit_down(
        &mut self,
        node: &'a ExprRef,
        context: Option<Field>,
    ) -> VortexResult<FoldDown<Self::Out, Self::Context>> {
        if let Some(item) = node.as_any().downcast_ref::<GetItem>() {
            return Ok(FoldDown::Continue(Some(item.field().clone())));
        };

        Ok(FoldDown::Continue(context))
    }

    fn visit_up(
        &mut self,
        node: &'a ExprRef,
        context: Option<Field>,
        children: FoldChildren<FieldAccesses>,
    ) -> VortexResult<FoldUp<FieldAccesses>> {
        let field_access = if node.as_any().downcast_ref::<Identity>().is_some() {
            assert!(children.is_empty());
            match context {
                Some(field) => FieldAccesses::Fields(HashSet::from_iter(iter::once(field))),
                None => FieldAccesses::AllFields,
            }
        } else {
            children
                .into_iter()
                .fold(FieldAccesses::Fields(HashSet::new()), |acc, x| {
                    acc.union(&x)
                })
        };

        // self.sub_expressions.insert(&node, field_access.clone());

        Ok(FoldUp::Continue(field_access))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traversal::Node;
    use crate::{get_item, ident};

    #[test]
    fn test_expr_top_level_ref() {
        let expr = ident();
        let mut expr_top_level_ref = ExprTopLevelRef::new();
        let res = expr
            .accept_with_context(&mut expr_top_level_ref, None)
            .unwrap();

        println!("{:?}", res);

        // let mut expected = HashMap::new();
        // expected.insert(&expr, FieldAccesses::AllFields);

        // assert_eq!(&expr_top_level_ref.sub_expressions, &expected);
    }

    #[test]
    fn test_expr_top_level_ref_get_item() {
        let expr = get_item("a", ident());
        let mut expr_top_level_ref = ExprTopLevelRef::new();
        let res = expr
            .accept_with_context(&mut expr_top_level_ref, None)
            .unwrap();

        println!("{:?}", res);

        // let mut expected = HashMap::new();
        // expected.insert(&expr, FieldAccesses::AllFields);

        // assert_eq!(&expr_top_level_ref.sub_expressions, &expected);
    }
}
