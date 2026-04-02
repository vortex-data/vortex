// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools as _;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray as _;
use crate::arrays::StructArray;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::Expression;
use crate::expr::lit;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::pack::Pack;
use crate::scalar_fn::fns::pack::PackOptions;
use crate::validity::Validity;

/// Merge zero or more expressions that ALL return structs.
///
/// If any field names are duplicated, the field from later expressions wins.
///
/// NOTE: Fields are not recursively merged, i.e. the later field REPLACES the earlier field.
/// This makes struct fields behaviour consistent with other dtypes.
#[derive(Clone)]
pub struct Merge;

impl ScalarFnVTable for Merge {
    type Options = DuplicateHandling;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new_ref("vortex.merge")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(match instance {
            DuplicateHandling::RightMost => vec![0x00],
            DuplicateHandling::Error => vec![0x01],
        }))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let instance = match _metadata {
            [0x00] => DuplicateHandling::RightMost,
            [0x01] => DuplicateHandling::Error,
            _ => {
                vortex_bail!("invalid metadata for Merge expression");
            }
        };
        Ok(instance)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Variadic { min: 0, max: None }
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        ChildName::from(Arc::from(format!("{}", child_idx)))
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "merge(")?;
        for (i, child) in expr.children().iter().enumerate() {
            child.fmt_sql(f)?;
            if i + 1 < expr.children().len() {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let mut field_names = Vec::new();
        let mut arrays = Vec::new();
        let mut merge_nullability = Nullability::NonNullable;
        let mut duplicate_names = HashSet::<_>::new();

        for dtype in arg_dtypes {
            let Some(fields) = dtype.as_struct_fields_opt() else {
                vortex_bail!("merge expects struct input");
            };
            if dtype.is_nullable() {
                vortex_bail!("merge expects non-nullable input");
            }

            merge_nullability |= dtype.nullability();

            for (field_name, field_dtype) in fields.names().iter().zip_eq(fields.fields()) {
                if let Some(idx) = field_names.iter().position(|name| name == field_name) {
                    duplicate_names.insert(field_name.clone());
                    arrays[idx] = field_dtype;
                } else {
                    field_names.push(field_name.clone());
                    arrays.push(field_dtype);
                }
            }
        }

        if options == &DuplicateHandling::Error && !duplicate_names.is_empty() {
            vortex_bail!(
                "merge: duplicate fields in children: {}",
                duplicate_names.into_iter().format(", ")
            )
        }

        Ok(DType::Struct(
            StructFields::new(FieldNames::from(field_names), arrays),
            merge_nullability,
        ))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Collect fields in order of appearance. Later fields overwrite earlier fields.
        let mut field_names = Vec::new();
        let mut arrays = Vec::new();
        let mut duplicate_names = HashSet::<_>::new();

        for i in 0..args.num_inputs() {
            let array = args.get(i)?.execute::<StructArray>(ctx)?;
            if array.dtype().is_nullable() {
                vortex_bail!("merge expects non-nullable input");
            }

            for (field_name, field_array) in array
                .names()
                .iter()
                .zip_eq(array.iter_unmasked_fields().cloned())
            {
                // Update or insert field.
                if let Some(idx) = field_names.iter().position(|name| name == field_name) {
                    duplicate_names.insert(field_name.clone());
                    arrays[idx] = field_array;
                } else {
                    field_names.push(field_name.clone());
                    arrays.push(field_array);
                }
            }
        }

        if options == &DuplicateHandling::Error && !duplicate_names.is_empty() {
            vortex_bail!(
                "merge: duplicate fields in children: {}",
                duplicate_names.into_iter().format(", ")
            )
        }

        // TODO(DK): When children are allowed to be nullable, this needs to change.
        let validity = Validity::NonNullable;
        let len = args.row_count();
        Ok(
            StructArray::try_new(FieldNames::from(field_names), arrays, len, validity)?
                .into_array(),
        )
    }

    fn reduce(
        &self,
        options: &Self::Options,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        let mut names = Vec::with_capacity(node.child_count() * 2);
        let mut children = Vec::with_capacity(node.child_count() * 2);
        let mut duplicate_names = HashSet::<_>::new();

        for child in (0..node.child_count()).map(|i| node.child(i)) {
            let child_dtype = child.node_dtype()?;
            if !child_dtype.is_struct() {
                vortex_bail!(
                    "Merge child must return a non-nullable struct dtype, got {}",
                    child_dtype
                )
            }

            let child_dtype = child_dtype
                .as_struct_fields_opt()
                .vortex_expect("expected struct");

            for name in child_dtype.names().iter() {
                if let Some(idx) = names.iter().position(|n| n == name) {
                    duplicate_names.insert(name.clone());
                    children[idx] = child.clone();
                } else {
                    names.push(name.clone());
                    children.push(child.clone());
                }
            }

            if options == &DuplicateHandling::Error && !duplicate_names.is_empty() {
                vortex_bail!(
                    "merge: duplicate fields in children: {}",
                    duplicate_names.into_iter().format(", ")
                )
            }
        }

        let pack_children: Vec<_> = names
            .iter()
            .zip(children)
            .map(|(name, child)| ctx.new_node(GetItem.bind(name.clone()), &[child]))
            .try_collect()?;

        let pack_expr = ctx.new_node(
            Pack.bind(PackOptions {
                names: FieldNames::from(names),
                nullability: node.node_dtype()?.nullability(),
            }),
            &pack_children,
        )?;

        Ok(Some(pack_expr))
    }

    fn validity(
        &self,
        _options: &Self::Options,
        _expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(lit(true)))
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, instance: &Self::Options) -> bool {
        matches!(instance, DuplicateHandling::Error)
    }
}

/// What to do when merged structs share a field name.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DuplicateHandling {
    /// If two structs share a field name, take the value from the right-most struct.
    RightMost,
    /// If two structs share a field name, error.
    #[default]
    Error,
}

impl Display for DuplicateHandling {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DuplicateHandling::RightMost => write!(f, "RightMost"),
            DuplicateHandling::Error => write!(f, "Error"),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType::I32;
    use crate::dtype::PType::I64;
    use crate::dtype::PType::U32;
    use crate::dtype::PType::U64;
    use crate::expr::Expression;
    use crate::expr::get_item;
    use crate::expr::merge;
    use crate::expr::merge_opts;
    use crate::expr::root;
    use crate::scalar_fn::fns::merge::DuplicateHandling;
    use crate::scalar_fn::fns::merge::StructArray;
    use crate::scalar_fn::fns::pack::Pack;

    fn primitive_field(array: &ArrayRef, field_path: &[&str]) -> VortexResult<PrimitiveArray> {
        let mut field_path = field_path.iter();

        let Some(field) = field_path.next() else {
            vortex_bail!("empty field path");
        };

        let mut array = array.to_struct().unmasked_field_by_name(field)?.clone();
        for field in field_path {
            array = array.to_struct().unmasked_field_by_name(field)?.clone();
        }
        Ok(array.to_primitive())
    }

    #[test]
    pub fn test_merge_right_most() {
        let expr = merge_opts(
            vec![
                get_item("0", root()),
                get_item("1", root()),
                get_item("2", root()),
            ],
            DuplicateHandling::RightMost,
        );

        let test_array = StructArray::from_fields(&[
            (
                "0",
                StructArray::from_fields(&[
                    ("a", buffer![0, 0, 0].into_array()),
                    ("b", buffer![1, 1, 1].into_array()),
                ])
                .unwrap()
                .into_array(),
            ),
            (
                "1",
                StructArray::from_fields(&[
                    ("b", buffer![2, 2, 2].into_array()),
                    ("c", buffer![3, 3, 3].into_array()),
                ])
                .unwrap()
                .into_array(),
            ),
            (
                "2",
                StructArray::from_fields(&[
                    ("d", buffer![4, 4, 4].into_array()),
                    ("e", buffer![5, 5, 5].into_array()),
                ])
                .unwrap()
                .into_array(),
            ),
        ])
        .unwrap()
        .into_array();
        let actual_array = test_array.apply(&expr).unwrap();

        assert_eq!(
            actual_array.as_struct_typed().names(),
            ["a", "b", "c", "d", "e"]
        );

        assert_arrays_eq!(
            primitive_field(&actual_array, &["a"]).unwrap(),
            PrimitiveArray::from_iter([0i32, 0, 0])
        );
        assert_arrays_eq!(
            primitive_field(&actual_array, &["b"]).unwrap(),
            PrimitiveArray::from_iter([2i32, 2, 2])
        );
        assert_arrays_eq!(
            primitive_field(&actual_array, &["c"]).unwrap(),
            PrimitiveArray::from_iter([3i32, 3, 3])
        );
        assert_arrays_eq!(
            primitive_field(&actual_array, &["d"]).unwrap(),
            PrimitiveArray::from_iter([4i32, 4, 4])
        );
        assert_arrays_eq!(
            primitive_field(&actual_array, &["e"]).unwrap(),
            PrimitiveArray::from_iter([5i32, 5, 5])
        );
    }

    #[test]
    #[should_panic(expected = "merge: duplicate fields in children")]
    pub fn test_merge_error_on_dupe_return_dtype() {
        let expr = merge_opts(
            vec![get_item("0", root()), get_item("1", root())],
            DuplicateHandling::Error,
        );
        let test_array = StructArray::try_from_iter([
            (
                "0",
                StructArray::try_from_iter([("a", buffer![1]), ("b", buffer![1])]).unwrap(),
            ),
            (
                "1",
                StructArray::try_from_iter([("c", buffer![1]), ("b", buffer![1])]).unwrap(),
            ),
        ])
        .unwrap()
        .into_array();

        expr.return_dtype(test_array.dtype()).unwrap();
    }

    #[test]
    #[should_panic(expected = "merge: duplicate fields in children")]
    pub fn test_merge_error_on_dupe_evaluate() {
        let expr = merge_opts(
            vec![get_item("0", root()), get_item("1", root())],
            DuplicateHandling::Error,
        );
        let test_array = StructArray::try_from_iter([
            (
                "0",
                StructArray::try_from_iter([("a", buffer![1]), ("b", buffer![1])]).unwrap(),
            ),
            (
                "1",
                StructArray::try_from_iter([("c", buffer![1]), ("b", buffer![1])]).unwrap(),
            ),
        ])
        .unwrap()
        .into_array();

        test_array.apply(&expr).unwrap();
    }

    #[test]
    pub fn test_empty_merge() {
        let expr = merge(Vec::<Expression>::new());

        let test_array = StructArray::from_fields(&[("a", buffer![0, 1, 2].into_array())])
            .unwrap()
            .into_array();
        let actual_array = test_array.clone().apply(&expr).unwrap();
        assert_eq!(actual_array.len(), test_array.len());
        assert_eq!(actual_array.as_struct_typed().nfields(), 0);
    }

    #[test]
    pub fn test_nested_merge() {
        // Nested structs are not merged!

        let expr = merge_opts(
            vec![get_item("0", root()), get_item("1", root())],
            DuplicateHandling::RightMost,
        );

        let test_array = StructArray::from_fields(&[
            (
                "0",
                StructArray::from_fields(&[(
                    "a",
                    StructArray::from_fields(&[
                        ("x", buffer![0, 0, 0].into_array()),
                        ("y", buffer![1, 1, 1].into_array()),
                    ])
                    .unwrap()
                    .into_array(),
                )])
                .unwrap()
                .into_array(),
            ),
            (
                "1",
                StructArray::from_fields(&[(
                    "a",
                    StructArray::from_fields(&[("x", buffer![0, 0, 0].into_array())])
                        .unwrap()
                        .into_array(),
                )])
                .unwrap()
                .into_array(),
            ),
        ])
        .unwrap()
        .into_array();
        let actual_array = test_array.apply(&expr).unwrap().to_struct();

        assert_eq!(
            actual_array
                .unmasked_field_by_name("a")
                .unwrap()
                .to_struct()
                .names()
                .iter()
                .map(|name| name.as_ref())
                .collect::<Vec<_>>(),
            vec!["x"]
        );
    }

    #[test]
    pub fn test_merge_order() {
        let expr = merge(vec![get_item("0", root()), get_item("1", root())]);

        let test_array = StructArray::from_fields(&[
            (
                "0",
                StructArray::from_fields(&[
                    ("a", buffer![0, 0, 0].into_array()),
                    ("c", buffer![1, 1, 1].into_array()),
                ])
                .unwrap()
                .into_array(),
            ),
            (
                "1",
                StructArray::from_fields(&[
                    ("b", buffer![2, 2, 2].into_array()),
                    ("d", buffer![3, 3, 3].into_array()),
                ])
                .unwrap()
                .into_array(),
            ),
        ])
        .unwrap()
        .into_array();
        let actual_array = test_array.apply(&expr).unwrap().to_struct();

        assert_eq!(actual_array.names(), ["a", "c", "b", "d"]);
    }

    #[test]
    pub fn test_display() {
        let expr = merge([get_item("struct1", root()), get_item("struct2", root())]);
        assert_eq!(expr.to_string(), "merge($.struct1, $.struct2)");

        let expr2 = merge(vec![get_item("a", root())]);
        assert_eq!(expr2.to_string(), "merge($.a)");
    }

    #[test]
    fn test_remove_merge() {
        let dtype = DType::struct_(
            [
                ("0", DType::struct_([("a", I32), ("b", I64)], NonNullable)),
                ("1", DType::struct_([("b", U32), ("c", U64)], NonNullable)),
            ],
            NonNullable,
        );

        let e = merge_opts(
            [get_item("0", root()), get_item("1", root())],
            DuplicateHandling::RightMost,
        );

        let result = e.optimize(&dtype).unwrap();

        assert!(result.is::<Pack>());
        assert_eq!(
            result.return_dtype(&dtype).unwrap(),
            DType::struct_([("a", I32), ("b", U32), ("c", U64)], NonNullable)
        );
    }
}
