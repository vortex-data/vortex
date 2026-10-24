// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Computing array statistics through the registered aggregate functions.

use std::sync::LazyLock;

use enum_iterator::all;
use itertools::Itertools;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

#[test]
fn test_uncompressed_size_in_bytes() -> VortexResult<()> {
    let struct_array = StructArray::new(
        FieldNames::from(["integers"]),
        vec![ConstantArray::new(5, 1000).into_array()],
        1000,
        Validity::NonNullable,
    );

    let canonical_size = struct_array
        .clone()
        .into_array()
        .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
        .into_array()
        .nbytes();
    let uncompressed_size = struct_array
        .statistics()
        .compute_uncompressed_size_in_bytes(&mut SESSION.create_execution_ctx());

    assert_eq!(canonical_size, 2);
    assert_eq!(uncompressed_size, Some(4000));
    Ok(())
}

#[test]
fn merge_unordered() {
    let array =
        PrimitiveArray::from_option_iter([Some(1), None, Some(2), Some(42), Some(10000), None]);
    let all_stats = all::<Stat>()
        .filter(|s| !matches!(s, Stat::Sum))
        .filter(|s| !matches!(s, Stat::NaNCount))
        .collect_vec();
    array
        .statistics()
        .compute_all(&all_stats, &mut SESSION.create_execution_ctx())
        .unwrap();

    let stats = array.statistics().to_owned();
    for stat in &all_stats {
        assert!(!stats.get(*stat).is_absent(), "Stat {stat} is missing");
    }

    let merged = stats.clone().merge_unordered(
        &stats,
        &DType::Primitive(PType::I32, Nullability::NonNullable),
    );
    for (stat, expected) in [
        (Stat::Min, Precision::exact(ScalarValue::from(1i32))),
        (Stat::Max, Precision::exact(ScalarValue::from(10000i32))),
        (Stat::NullCount, Precision::exact(ScalarValue::from(4u64))),
        (
            Stat::UncompressedSizeInBytes,
            Precision::exact(ScalarValue::from(50u64)),
        ),
        (Stat::IsConstant, Precision::exact(ScalarValue::from(false))),
        (Stat::IsSorted, Precision::Absent),
        (Stat::IsStrictSorted, Precision::Absent),
    ] {
        assert_eq!(merged.get(stat), expected, "{stat}");
    }
}
