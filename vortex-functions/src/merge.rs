// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools as _;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray as _;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ReduceNode;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::get_item::GetItem;
use vortex_array::scalar_fn::builtin::pack::Pack;
use vortex_array::scalar_fn::builtin::pack::PackOptions;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_set::HashSet;

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
        static ID: CachedId = CachedId::new("vortex.merge");
        *ID
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

    fn reduce<T: ReduceNode>(&self, options: &Self::Options, node: &T) -> VortexResult<Option<T>> {
        let mut names = Vec::with_capacity(node.child_count() * 2);
        let mut fields: Vec<T> = Vec::with_capacity(node.child_count() * 2);
        let mut duplicate_names = HashSet::<_>::new();

        for child in (0..node.child_count()).map(|i| node.child(i)) {
            let child_dtype = child.node_dtype()?;
            if !child_dtype.is_struct() || child_dtype.is_nullable() {
                vortex_bail!(
                    "Merge child must return a non-nullable struct dtype, got {}",
                    child_dtype
                )
            }

            let child_dtype = child_dtype
                .as_struct_fields_opt()
                .vortex_expect("expected struct");

            for name in child_dtype.names().iter() {
                // `apply_now` reads straight out of a `pack` child, so this reduction leaves
                // behind no `get_item` for a later pass to simplify. The child is non-nullable,
                // so the read cannot intersect a struct validity into the field, which is the
                // only way it could change the field dtype.
                let field = GetItem::apply_now(&child, name)?;

                if let Some(idx) = names.iter().position(|n| n == name) {
                    duplicate_names.insert(name.clone());
                    fields[idx] = field;
                } else {
                    names.push(name.clone());
                    fields.push(field);
                }
            }

            if options == &DuplicateHandling::Error && !duplicate_names.is_empty() {
                vortex_bail!(
                    "merge: duplicate fields in children: {}",
                    duplicate_names.into_iter().format(", ")
                )
            }
        }

        let pack_expr = node.new_node(
            Pack.bind(PackOptions {
                names: FieldNames::from(names),
                nullability: node.node_dtype()?.nullability(),
            }),
            &fields,
        )?;

        Ok(Some(pack_expr))
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, instance: &Self::Options) -> bool {
        !matches!(instance, DuplicateHandling::Error)
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
    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType::I32;
    use vortex_array::dtype::PType::I64;
    use vortex_array::dtype::PType::U32;
    use vortex_array::dtype::PType::U64;
    use vortex_array::dtype::StructFields;
    use vortex_array::scalar_fn::builtin::pack::Pack;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::BoundExpression;
    use vortex_expr::Expression;
    use vortex_expr::analysis::make_bound_free_field_annotator;
    use vortex_expr::col;
    use vortex_expr::get_item;
    use vortex_expr::pack;
    use vortex_expr::root;
    use vortex_expr::transform::partition_bound;
    use vortex_session::VortexSession;

    use crate::expr::merge;
    use crate::expr::merge_opts;
    use crate::merge::DuplicateHandling;
    use crate::merge::StructArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();

        crate::initialize(&session);

        session
    });

    fn field_array(array: &ArrayRef, field_path: &[&str]) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut field_path = field_path.iter();

        let Some(field) = field_path.next() else {
            vortex_bail!("empty field path");
        };

        let mut array = array
            .clone()
            .execute::<StructArray>(&mut ctx)?
            .unmasked_field_by_name(field)?
            .clone();
        for field in field_path {
            let next = array
                .clone()
                .execute::<StructArray>(&mut ctx)?
                .unmasked_field_by_name(field)?
                .clone();
            array = next;
        }
        Ok(array)
    }

    #[test]
    pub fn test_merge_right_most() {
        let mut ctx = SESSION.create_execution_ctx();
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
            actual_array.dtype().as_struct_fields().names(),
            ["a", "b", "c", "d", "e"]
        );

        assert_arrays_eq!(
            field_array(&actual_array, &["a"]).unwrap(),
            PrimitiveArray::from_iter([0i32, 0, 0]),
            &mut ctx
        );
        assert_arrays_eq!(
            field_array(&actual_array, &["b"]).unwrap(),
            PrimitiveArray::from_iter([2i32, 2, 2]),
            &mut ctx
        );
        assert_arrays_eq!(
            field_array(&actual_array, &["c"]).unwrap(),
            PrimitiveArray::from_iter([3i32, 3, 3]),
            &mut ctx
        );
        assert_arrays_eq!(
            field_array(&actual_array, &["d"]).unwrap(),
            PrimitiveArray::from_iter([4i32, 4, 4]),
            &mut ctx
        );
        assert_arrays_eq!(
            field_array(&actual_array, &["e"]).unwrap(),
            PrimitiveArray::from_iter([5i32, 5, 5]),
            &mut ctx
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
        let expected = StructArray::try_new(
            Default::default(),
            vec![],
            3,
            vortex_array::validity::Validity::NonNullable,
        )
        .unwrap();
        assert_arrays_eq!(actual_array, expected, &mut SESSION.create_execution_ctx());
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
        let mut ctx = SESSION.create_execution_ctx();
        let actual_array = test_array
            .apply(&expr)
            .unwrap()
            .execute::<StructArray>(&mut ctx)
            .unwrap();

        let inner_struct = actual_array.unmasked_field_by_name("a").unwrap();
        assert_eq!(
            inner_struct
                .dtype()
                .as_struct_fields()
                .names()
                .iter()
                .map(|name| name.as_ref())
                .collect::<Vec<_>>(),
            vec!["x"]
        );
        assert_arrays_eq!(
            inner_struct,
            StructArray::from_fields(&[("x", buffer![0i32, 0, 0].into_array())]).unwrap(),
            &mut ctx
        );
    }

    #[test]
    pub fn test_merge_order() {
        let mut ctx = SESSION.create_execution_ctx();
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
        let actual_array = test_array.apply(&expr).unwrap();

        assert_eq!(
            actual_array.dtype().as_struct_fields().names(),
            ["a", "c", "b", "d"]
        );
        let expected = StructArray::from_fields(&[
            ("a", buffer![0i32, 0, 0].into_array()),
            ("c", buffer![1i32, 1, 1].into_array()),
            ("b", buffer![2i32, 2, 2].into_array()),
            ("d", buffer![3i32, 3, 3].into_array()),
        ])
        .unwrap();
        assert_arrays_eq!(actual_array, expected, &mut ctx);
    }

    #[test]
    pub fn test_display() {
        let expr = merge([get_item("struct1", root()), get_item("struct2", root())]);
        assert_eq!(
            expr.to_string(),
            "vortex.merge($.struct1, $.struct2, opts=Error)"
        );

        let expr2 = merge(vec![get_item("a", root())]);
        assert_eq!(expr2.to_string(), "vortex.merge($.a, opts=Error)");
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

        let result = e.bind(&dtype).unwrap().optimize().unwrap();

        assert!(result.is::<Pack>());
        assert_eq!(
            result.dtype(),
            &DType::struct_([("a", I32), ("b", U32), ("c", U64)], NonNullable)
        );
    }

    fn partition_root_dtype(names: &FieldNames, partitions: &[BoundExpression]) -> DType {
        DType::Struct(
            StructFields::new(
                names.clone(),
                partitions
                    .iter()
                    .map(|partition| partition.dtype().clone())
                    .collect(),
            ),
            NonNullable,
        )
    }

    #[test]
    fn partitions_merge_by_field() -> VortexResult<()> {
        let dtype = DType::Struct(
            StructFields::from_iter([
                (
                    "a",
                    DType::Struct(
                        StructFields::from_iter([("x", DType::from(I32)), ("y", DType::from(I32))]),
                        NonNullable,
                    ),
                ),
                ("b", DType::from(I32)),
                ("c", DType::from(I32)),
            ]),
            NonNullable,
        );
        let fields = dtype.as_struct_fields_opt().vortex_expect("struct");
        let expr = merge([col("a"), pack([("b", col("b"))], NonNullable)]);

        let partitioned =
            partition_bound(expr.bind(&dtype)?, make_bound_free_field_annotator(fields))?;
        let expected = merge([get_item("a_0", col("a")), get_item("b_0", col("b"))]);
        let root_dtype =
            partition_root_dtype(&partitioned.partition_names, &partitioned.partitions);
        assert_eq!(
            partitioned.root,
            expected.bind(&root_dtype)?,
            "{} {}",
            partitioned.root,
            expected
        );

        assert_eq!(partitioned.partitions.len(), 2);

        let part_a = partitioned
            .find_partition(&"a".into())
            .vortex_expect("partition a");
        let expected_a = pack([("a_0", col("a"))], NonNullable);
        assert_eq!(part_a, &expected_a.bind(&dtype)?, "{part_a} {expected_a}");

        let part_b = partitioned
            .find_partition(&"b".into())
            .vortex_expect("partition b");
        let expected_b = pack([("b_0", pack([("b", col("b"))], NonNullable))], NonNullable);
        assert_eq!(part_b, &expected_b.bind(&dtype)?, "{part_b} {expected_b}");
        Ok(())
    }
}
