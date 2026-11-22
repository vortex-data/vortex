// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType::I32;
use vortex_array::dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_expr::BoundExpression;
use vortex_expr::col;
use vortex_expr::get_item;
use vortex_expr::pack;
use vortex_expr::transform::partition_bound;
use vortex_functions::expr::merge;

use super::make_bound_free_field_annotator;

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

/// A `merge` reads each merged child from the partition of the field it accesses.
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

    let partitioned = partition_bound(expr.bind(&dtype)?, make_bound_free_field_annotator(fields))?;
    let expected = merge([get_item("a_0", col("a")), get_item("b_0", col("b"))]);
    let root_dtype = partition_root_dtype(&partitioned.partition_names, &partitioned.partitions);
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
