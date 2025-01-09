use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_dtype::{Field, FieldName, FieldNames};
use vortex_error::VortexResult;

use crate::traversal::{FoldChildren, FoldUp, Folder};
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

// #[derive(Clone, Debug, PartialEq, Eq)]
// enum FieldAccesses {
//     Fields(HashSet<Field>),
//     AllFields,
// }
//
// impl FieldAccesses {
//     fn union(self, other: &Self) -> Self {
//         match (self, other) {
//             (FieldAccesses::AllFields, _) => FieldAccesses::AllFields,
//             (_, FieldAccesses::AllFields) => FieldAccesses::AllFields,
//             (FieldAccesses::Fields(fields1), FieldAccesses::Fields(fields2)) => {
//                 FieldAccesses::Fields(fields1.union(fields2).cloned().collect())
//             }
//         }
//     }
// }

// For all subexpressions in an expression find the fields access directly on identity
struct ExprTopLevelRef<'a> {
    sub_expressions: HashMap<&'a ExprRef, HashSet<Field>>,
    identity: FieldNames,
}

impl<'a> ExprTopLevelRef<'a> {
    fn new(fields: FieldNames) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            identity: fields,
        }
    }
}

impl<'a> Folder<'a> for ExprTopLevelRef<'a> {
    type NodeTy = ExprRef;
    type Out = HashSet<FieldName>;
    type Context = ();

    fn visit_up(
        &mut self,
        node: &'a ExprRef,
        _context: (),
        children: FoldChildren<HashSet<FieldName>>,
    ) -> VortexResult<FoldUp<HashSet<FieldName>>> {
        if node.as_any().downcast_ref::<Identity>().is_some() {
            debug_assert!(children.is_empty());
            let field_names = HashSet::from_iter(self.identity.iter().cloned());
            return Ok(FoldUp::Continue(field_names));
        }

        if let Some(item) = node.as_any().downcast_ref::<GetItem>() {
            let field = item.field();
            let field_name = match field {
                Field::Name(n) => n.clone(),
                Field::Index(_) => todo!(),
            };
            // let [child] = children.unwrap().as_slice();
            // let access = child.get(&field_name).cloned();
            let field_access = HashSet::from_iter(vec![field.clone()]);
            self.sub_expressions.insert(&node, field_access.clone());
            return Ok(FoldUp::Continue(HashSet::from_iter(vec![field_name])));
        }

        // else {
        // };

        let field_access = children.into_iter().fold(HashSet::new(), |acc, fields| {
            acc.union(&fields).cloned().collect()
        });
        self.sub_expressions.insert(
            &node,
            field_access
                .clone()
                .iter()
                .map(|f| Field::Name(f.clone()))
                .collect(),
        );

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
        let mut expr_top_level_ref =
            ExprTopLevelRef::new(FieldNames::from_iter(vec!["a".into(), "b".into()]));
        let res = expr
            .accept_with_context(&mut expr_top_level_ref, ())
            .unwrap();

        println!("{:?}", res);
        println!("{:?}", expr_top_level_ref.sub_expressions);
    }

    #[test]
    fn test_expr_top_level_ref_get_item() {
        let expr = get_item("b", get_item("a", ident()));
        let mut expr_top_level_ref =
            ExprTopLevelRef::new(FieldNames::from_iter(vec!["a".into(), "b".into()]));
        let res = expr
            .accept_with_context(&mut expr_top_level_ref, ())
            .unwrap();

        println!("{:?}", res);
        println!("{:?}", expr_top_level_ref.sub_expressions);

        // let mut expected = HashMap::new();
        // expected.insert(&expr, FieldAccesses::AllFields);

        // assert_eq!(&expr_top_level_ref.sub_expressions, &expected);
    }
}
