// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex::aggregate_fn::AggregateFnRef;
use vortex::aggregate_fn::AggregateFnVTableExt;
use vortex::aggregate_fn::EmptyOptions;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::fns::is_sorted::IsSorted;
use vortex::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::nan_count::NanCount;
use vortex::aggregate_fn::fns::null_count::NullCount;
use vortex::aggregate_fn::fns::sum::Sum;
use vortex::aggregate_fn::fns::sum_v2::SumV2;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::validity::Validity;
use vortex::buffer::buffer;
use vortex::dtype::FieldNames;
use vortex::dtype::FieldPath;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;

use super::FileAggregate;
use super::aggregate_file;
use crate::RUNTIME;
use crate::session::new_session;

fn fixture(
    session: &VortexSession,
    array: ArrayRef,
    stats: Vec<AggregateFnRef>,
) -> VortexResult<VortexFile> {
    let mut bytes = Vec::new();
    RUNTIME.block_on(async {
        session
            .write_options()
            .disable_editions()
            .include_nested_statistics()
            .with_field_file_statistics(FieldPath::from_name("v"), stats)
            .write(&mut bytes, array.to_array_stream())
            .await
    })?;
    session.open_options().open_buffer(bytes)
}

fn query(
    file: &VortexFile,
    session: &VortexSession,
    tags: &[u8],
    scan: bool,
) -> VortexResult<Vec<Option<Scalar>>> {
    let requests = tags
        .iter()
        .map(|&tag| FileAggregate::from_tag(tag, Some("v".into())))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(aggregate_file(file, &requests, scan, session)?.1)
}

#[rstest]
#[case::ordered([1, 1, 2], true)]
#[case::unordered([1, 3, 2], false)]
fn sortedness_from_a_selected_field(
    #[case] values: [i32; 3],
    #[case] expected: bool,
) -> VortexResult<()> {
    let session = new_session();
    let array = StructArray::from_fields(&[("v", PrimitiveArray::from_iter(values).into_array())])?
        .into_array();
    let file = fixture(
        &session,
        array,
        vec![IsSorted.bind(IsSortedOptions { strict: false })],
    )?;
    assert_eq!(
        query(&file, &session, &[5], false)?[0]
            .as_ref()
            .and_then(|s| s.as_bool().value()),
        Some(expected)
    );
    Ok(())
}

#[test]
fn sortedness_requires_footer_evidence_for_nullable_floats() -> VortexResult<()> {
    let session = new_session();
    let array = StructArray::from_fields(&[(
        "v",
        PrimitiveArray::from_option_iter([Some(1.0), Some(2.0)]).into_array(),
    )])?
    .into_array();
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    let unknown = fixture(&session, array.clone(), vec![sorted.clone()])?;
    assert!(query(&unknown, &session, &[5], true)?[0].is_none());
    let known = fixture(
        &session,
        array,
        vec![
            sorted,
            NullCount.bind(EmptyOptions),
            NanCount.bind(EmptyOptions),
        ],
    )?;
    assert_eq!(
        query(&known, &session, &[5], false)?[0]
            .as_ref()
            .and_then(|s| s.as_bool().value()),
        Some(true)
    );
    Ok(())
}

#[test]
fn fallback_aggregates_respect_nullable_parent_rows() -> VortexResult<()> {
    let session = new_session();
    let array = StructArray::new(
        FieldNames::from(["v"]),
        [buffer![1i64, 99, -100, 4].into_array()],
        4,
        Validity::Array(BoolArray::from_iter([true, false, false, true]).into_array()),
    );
    let file = fixture(&session, array.into_array(), vec![])?;
    let tags = [1, 2, 3, 4, 5, 4];
    assert!(
        query(&file, &session, &tags, false)?
            .iter()
            .all(Option::is_none)
    );
    let answers = query(&file, &session, &tags, true)?;
    assert_eq!(answers[0], Some(Scalar::from(2u64)));
    for (index, expected) in [(1, 1), (2, 4), (3, 5), (5, 5)] {
        assert_eq!(
            answers[index]
                .as_ref()
                .and_then(|s| s.as_primitive().typed_value::<i64>()),
            Some(expected)
        );
    }
    assert!(answers[4].is_none());
    Ok(())
}

#[rstest]
#[case::finite_with_nan([Some(-1.5), Some(f64::NAN), None], Some(-1.5), Some(f64::NAN), 2)]
#[case::only_nans_and_nulls([Some(f64::NAN), None, None], Some(f64::NAN), Some(f64::NAN), 1)]
#[case::only_nulls([None; 3], None, None, 0)]
#[case::no_nans([Some(2.5), None, Some(-1.5)], Some(-1.5), Some(2.5), 2)]
fn float_extrema_order_nan_last(
    #[case] values: [Option<f64>; 3],
    #[case] min: Option<f64>,
    #[case] max: Option<f64>,
    #[case] count: u64,
) -> VortexResult<()> {
    let session = new_session();
    let array =
        StructArray::from_fields(&[("v", PrimitiveArray::from_option_iter(values).into_array())])?
            .into_array();
    for scan in [false, true] {
        let stats = if scan {
            vec![]
        } else {
            vec![
                Min.bind(NumericalAggregateOpts::default()),
                Max.bind(NumericalAggregateOpts::default()),
                NanCount.bind(EmptyOptions),
                NullCount.bind(EmptyOptions),
            ]
        };
        let file = fixture(&session, array.clone(), stats)?;
        let answers = query(&file, &session, &[2, 3, 1], scan)?;
        for (answer, expected) in answers[..2].iter().zip([min, max]) {
            let value = answer
                .as_ref()
                .expect("answered")
                .as_primitive()
                .typed_value::<f64>();
            assert_eq!(value.map(f64::to_bits), expected.map(f64::to_bits));
        }
        assert_eq!(
            answers[2]
                .as_ref()
                .and_then(|s| s.as_primitive().typed_value::<u64>()),
            Some(count)
        );
    }
    Ok(())
}

#[rstest]
#[case::all_nan([Some(f32::NAN), None], true)]
#[case::mixed([Some(f32::NAN), Some(1.0)], false)]
fn nan_count_can_prove_extrema_without_recorded_extrema(
    #[case] values: [Option<f32>; 2],
    #[case] min_is_known: bool,
) -> VortexResult<()> {
    let session = new_session();
    let array =
        StructArray::from_fields(&[("v", PrimitiveArray::from_option_iter(values).into_array())])?
            .into_array();
    let file = fixture(
        &session,
        array,
        vec![NanCount.bind(EmptyOptions), NullCount.bind(EmptyOptions)],
    )?;
    let footer = query(&file, &session, &[2, 3], false)?;
    assert_eq!(footer[0].is_some(), min_is_known);
    assert!(
        footer[1]
            .as_ref()
            .expect("answered maximum")
            .as_primitive()
            .is_nan()
    );
    let scanned = query(&file, &session, &[2, 3], true)?;
    let min = scanned[0]
        .as_ref()
        .expect("answered minimum")
        .as_primitive();
    if min_is_known {
        assert!(min.is_nan());
    } else {
        assert_eq!(min.typed_value::<f32>(), Some(1.0));
    }
    Ok(())
}

#[rstest]
#[case::empty([None, None], None)]
#[case::zero([Some(-1), Some(1)], Some(0))]
fn legacy_sum_needs_evidence_of_non_null_input(
    #[case] values: [Option<i64>; 2],
    #[case] expected: Option<i64>,
) -> VortexResult<()> {
    let session = new_session();
    let array =
        StructArray::from_fields(&[("v", PrimitiveArray::from_option_iter(values).into_array())])?
            .into_array();
    let sum = Sum.bind(NumericalAggregateOpts::default());
    let ambiguous = fixture(&session, array.clone(), vec![sum.clone()])?;
    assert!(query(&ambiguous, &session, &[4], false)?[0].is_none());
    let known = fixture(&session, array, vec![sum, NullCount.bind(EmptyOptions)])?;
    let answer = query(&known, &session, &[4], false)?;
    assert_eq!(
        answer[0]
            .as_ref()
            .expect("answered sum")
            .as_primitive()
            .typed_value::<i64>(),
        expected
    );
    Ok(())
}

#[test]
fn overflow_is_an_answered_null() -> VortexResult<()> {
    let session = new_session();
    let array = StructArray::from_fields(&[("v", buffer![i64::MAX, 1].into_array())])?.into_array();
    for scan in [false, true] {
        let stats = if scan {
            vec![]
        } else {
            vec![SumV2.bind(NumericalAggregateOpts::default())]
        };
        let file = fixture(&session, array.clone(), stats)?;
        assert!(
            query(&file, &session, &[4], scan)?[0]
                .as_ref()
                .is_some_and(Scalar::is_null)
        );
    }
    Ok(())
}
