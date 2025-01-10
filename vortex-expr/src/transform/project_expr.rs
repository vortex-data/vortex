use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_dtype::DType::Struct;
use vortex_dtype::{Field, FieldName, FieldNames, StructDType};
use vortex_error::VortexResult;

use crate::traversal::{FoldChildren, FoldDown, FoldUp, Folder, FolderMut, Node};
use crate::{get_item, ident, pack, ExprRef, GetItem, Identity, Select, SelectField};

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

// struct ExprSplitter {
//     field: Field,
//     sub_expressions: Vec<ExprRef>,
// }
//
// impl ExprSplitter {
//     fn new(field: Field) -> Self {
//         Self {
//             field,
//             sub_expressions: vec![],
//         }
//     }
// }

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
// enum AccessibleFields {
//     Fields(HashSet<FieldName>),
//     UntrackedFields,
// }
//
// impl AccessibleFields {
//     fn important_fields(&self) -> HashSet<FieldName> {
//         match self {
//             AccessibleFields::Fields(fields) => fields.clone(),
//             AccessibleFields::UntrackedFields => HashSet::new(),
//         }
//     }
//
//     fn get_field(&self, field: &FieldName) -> Option<FieldName> {
//         match self {
//             AccessibleFields::Fields(fields) => fields.get(field).cloned(),
//             AccessibleFields::UntrackedFields => None,
//         }
//     }
// }
//

type FieldAccesses<'a> = HashMap<&'a ExprRef, HashSet<Field>>;

// For all subexpressions in an expression find the fields access directly on identity
struct ExprTopLevelRef<'a> {
    sub_expressions: FieldAccesses<'a>,
    ident_dt: StructDType,
}

impl<'a> ExprTopLevelRef<'a> {
    fn new(ident_dt: StructDType) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            ident_dt,
        }
    }
}

// This is a very naive, but simple analysis to find the fields that are accessed directly on an
// identity node. This is combined to provide an over-approximation of the fields that are accessed
// by an expression.
impl<'a> Folder<'a> for ExprTopLevelRef<'a> {
    type NodeTy = ExprRef;
    type Out = ();
    type Context = ();

    fn visit_down(
        &mut self,
        node: &'a Self::NodeTy,
        _context: (),
    ) -> VortexResult<FoldDown<Self::Out, Self::Context>> {
        if let Some(get_item) = node.as_any().downcast_ref::<GetItem>() {
            if get_item
                .child()
                .as_any()
                .downcast_ref::<Identity>()
                .is_some()
            {
                self.sub_expressions
                    .insert(&node, HashSet::from_iter(vec![get_item.field().clone()]));

                return Ok(FoldDown::SkipChildren);
            }
        } else if let Some(select) = node.as_any().downcast_ref::<Select>() {
            assert!(matches!(select.fields(), SelectField::Include(_)));
            if select.child().as_any().downcast_ref::<Identity>().is_some() {
                self.sub_expressions.insert(
                    &node,
                    HashSet::from_iter(select.fields().fields().iter().cloned()),
                );
            }
            return Ok(FoldDown::SkipChildren);
        } else if node.as_any().downcast_ref::<Identity>().is_some() {
            let st_dtype = &self.ident_dt;
            self.sub_expressions.insert(
                &node,
                st_dtype
                    .names()
                    .iter()
                    .map(|n| Field::Name(n.clone()))
                    .collect(),
            );
            self.sub_expressions.insert(
                &node,
                st_dtype
                    .names()
                    .iter()
                    .cloned()
                    .map(|n| Field::Name(n))
                    .collect(),
            );
        }

        Ok(FoldDown::Continue(()))
    }

    fn visit_up(
        &mut self,
        node: &'a ExprRef,
        _context: (),
        _children: FoldChildren<()>,
    ) -> VortexResult<FoldUp<()>> {
        let accesses = node
            .children()
            .iter()
            .map(|c| self.sub_expressions.get(c).cloned())
            .collect_vec();

        accesses.into_iter().for_each(|c| {
            if let Some(fields) = c {
                self.sub_expressions
                    .entry(node)
                    .or_insert_with(HashSet::new)
                    .extend(fields.iter().cloned());
            }
        });

        Ok(FoldUp::Continue(()))
    }
}

#[derive(Debug)]
struct ExprSplitter<'a> {
    sub_expressions: HashMap<Field, Vec<ExprRef>>,
    accesses: FieldAccesses<'a>,
    dt_ident: StructDType,
}

impl<'a> ExprSplitter<'a> {
    fn new(accesses: FieldAccesses<'a>, dt_ident: StructDType) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            accesses,
            dt_ident,
        }
    }

    fn new_expr_name(f: &Field) -> FieldName {
        let name = match f {
            Field::Name(n) => n.clone(),
            Field::Index(i) => i.to_string().into(),
        };
        format!("__e_{}", name).into()
    }

    fn split(
        expr: ExprRef,
        ident_dt: StructDType,
    ) -> VortexResult<(ExprRef, HashMap<Field, (FieldName, ExprRef)>)> {
        let mut expr_top_level_ref = ExprTopLevelRef::new(ident_dt.clone());
        expr.accept_with_context(&mut expr_top_level_ref, ())?;

        let mut splitter = ExprSplitter::new(expr_top_level_ref.sub_expressions, ident_dt);

        let split = expr.clone().transform_with_context(&mut splitter, ())?;

        Ok((
            split.result(),
            splitter
                .sub_expressions
                .into_iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        (
                            Self::new_expr_name(&k),
                            pack(
                                (0..v.len())
                                    .into_iter()
                                    .map(|i| FieldName::from(i.to_string()))
                                    .collect_vec(),
                                v,
                            ),
                        ),
                    )
                })
                .collect(),
        ))
    }
}

impl<'a> FolderMut for ExprSplitter<'a> {
    type NodeTy = ExprRef;
    type Out = ExprRef;
    type Context = ();

    fn visit_down(
        &mut self,
        node: &Self::NodeTy,
        _context: Self::Context,
    ) -> VortexResult<FoldDown<ExprRef, Self::Context>> {
        if self
            .accesses
            .get(node)
            .map(|a| a.len() == 1)
            .unwrap_or(false)
        {
            // Found the top most sub-expression which only access a single field
            return Ok(FoldDown::SkipChildren);
        };

        Ok(FoldDown::Continue(()))
    }

    fn visit_up(
        &mut self,
        node: Self::NodeTy,
        _context: Self::Context,
        children: FoldChildren<ExprRef>,
    ) -> VortexResult<FoldUp<ExprRef>> {
        if self
            .accesses
            .get(&node)
            .map(|a| a.len() == 1)
            .unwrap_or(false)
        {
            let field = self.accesses.get(&node).unwrap().iter().next().unwrap();
            let sub_exprs = self
                .sub_expressions
                .entry(field.clone())
                .or_insert_with(Vec::new);
            let idx = sub_exprs.len();

            let fname = match field {
                Field::Name(n) => n.clone(),
                Field::Index(i) => self.dt_ident.names()[*i].clone(),
            };

            // Need to replace get_item(f, ident) with ident, making the expr relative to the child.
            let replaced = node.transform_with_context(&mut ExprReplaceFieldAccess(fname), ())?;
            sub_exprs.push(replaced.result());

            let access = get_item(
                idx,
                get_item(Field::Name(Self::new_expr_name(field)), ident()),
            );

            return Ok(FoldUp::Continue(access));
        };

        if node.as_any().downcast_ref::<Identity>().is_some() {
            let fields = (0..self.dt_ident.names().len())
                .map(|f| Field::Index(f))
                .collect_vec();

            for f in &fields {
                self.sub_expressions
                    .entry(f.clone())
                    .or_insert_with(Vec::new)
                    .push(ident())
            }

            let pack_expr = pack(
                fields.iter().map(|f| Self::new_expr_name(f)).collect_vec(),
                fields.into_iter().map(|_| ident()).collect(),
            );

            return Ok(FoldUp::Continue(pack_expr));
        }

        match children {
            FoldChildren::Skipped => unreachable!("Children skipped handled above"),
            FoldChildren::Children(c) => Ok(FoldUp::Continue(node.replacing_children(c))),
        }
    }
}

struct ExprReplaceFieldAccess(FieldName);

impl FolderMut for ExprReplaceFieldAccess {
    type NodeTy = ExprRef;
    type Out = ExprRef;
    type Context = ();

    fn visit_up(
        &mut self,
        node: Self::NodeTy,
        _context: (),
        children: FoldChildren<Self::Out>,
    ) -> VortexResult<FoldUp<Self::Out>> {
        if node.as_any().downcast_ref::<Identity>().is_some() {
            Ok(FoldUp::Continue(pack(vec![self.0.clone()], vec![ident()])))
        } else {
            assert!(!matches!(children, FoldChildren::Skipped));
            Ok(FoldUp::Continue(
                node.replacing_children(children.into_iter().collect()),
            ))
        }
    }
}

#[cfg(test)]
mod tests {

    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::StructDType;

    use super::*;
    use crate::transform::expr_simplify::ExprSimplify;
    use crate::{add, get_item, ident, lit, pack, Pack};

    fn struct_dtype() -> StructDType {
        StructDType::new(
            vec!["a".into(), "b".into(), "c".into()].into(),
            vec![
                Struct(
                    StructDType::new(
                        vec!["a".into(), "b".into()].into(),
                        vec![I32.into(), I32.into()].into(),
                    ),
                    NonNullable,
                ),
                I32.into(),
                I32.into(),
            ]
            .into(),
        )
    }

    #[test]
    fn test_expr_top_level_ref() {
        let dtype = struct_dtype();

        let expr = ident();

        let split = ExprSplitter::split(expr, dtype.clone());

        assert!(split.is_ok());

        let (top, splits) = split.unwrap();

        assert!(top.as_any().downcast_ref::<Pack>().is_some());
        // Have a single top level pack with all fields in dtype
        assert_eq!(splits.len(), dtype.names().len())
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split() {
        let dtype = struct_dtype();

        let expr = get_item("b", get_item("a", ident()));

        let (top, splits) = ExprSplitter::split(expr, dtype.clone()).unwrap();

        let split_a = splits.get(&Field::Name("a".into()));
        assert!(split_a.is_some());
        let split_a = split_a.unwrap();

        assert_eq!(
            &top,
            &get_item(0, get_item(Field::Name(split_a.0.clone()), ident()))
        );
        assert_eq!(
            &ExprSimplify::simplify(split_a.1.clone()).unwrap(),
            &pack(vec!["0".into()], vec![get_item("b", ident())])
        );
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split_pack() {
        let dtype = struct_dtype();

        let expr = pack(
            vec!["a".into(), "b".into(), "c".into()],
            vec![
                get_item("a", get_item("a", ident())),
                get_item("b", get_item("a", ident())),
                get_item("c", ident()),
            ],
        );
        let (_, splits) = ExprSplitter::split(expr, dtype).unwrap();

        let split_a = splits.get(&Field::Name("a".into())).unwrap();
        assert_eq!(
            &ExprSimplify::simplify(split_a.1.clone()).unwrap(),
            &pack(
                vec!["0".into(), "1".into()],
                vec![get_item("a", ident()), get_item("b", ident())]
            )
        );
        let split_c = splits.get(&Field::Name("c".into())).unwrap();
        assert_eq!(
            &ExprSimplify::simplify(split_c.1.clone()).unwrap(),
            &pack(vec!["0".into()], vec![ident()])
        )
    }

    #[test]
    fn test_expr_top_level_ref_get_item_add() {
        let dtype = struct_dtype();

        let expr = add(get_item("b", get_item("a", ident())), lit(1));
        let (_, splits) = ExprSplitter::split(expr, dtype).unwrap();

        // Whole expr is a single split
        assert_eq!(splits.len(), 1);
    }

    #[test]
    fn test_expr_top_level_ref_get_item_add_cannot_split() {
        let dtype = struct_dtype();

        let expr = add(
            get_item("b", get_item("a", ident())),
            get_item("b", ident()),
        );
        let (_, splits) = ExprSplitter::split(expr, dtype).unwrap();

        // One for id.a and id.b
        assert_eq!(splits.len(), 2);
    }
}
