// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use rstest::rstest;
use rustc_hash::FxBuildHasher;
use vortex_array::Canonical;
use vortex_array::Columnar;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateDTypes;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use super::float::FloatDistinct;
use super::float::FloatDistinctPartial;
use super::integer::IntegerFrequencies;
use super::integer::IntegerFrequenciesPartial;

/// Run one native distribution batch through its aggregate contract.
fn summarize<V>(vtable: &V, array: PrimitiveArray) -> VortexResult<V::Partial>
where
    V: AggregateFnVTable<Options = EmptyOptions>,
{
    let dtypes = AggregateDTypes::try_new(vtable, &EmptyOptions, array.dtype().clone())?;
    let args = dtypes.args(&EmptyOptions);
    let mut partial = vtable.empty_partial(args)?;
    let mut ctx = array_session().create_execution_ctx();
    vtable.accumulate(
        args,
        &mut partial,
        &Columnar::Canonical(Canonical::Primitive(array)),
        &mut ctx,
    )?;
    Ok(partial)
}

/// Count valid values independently of the dense and run-based kernels.
fn reference_frequencies<T>(
    values: &[T],
    validity: &[bool],
) -> HashMap<NativeValue<T>, u32, FxBuildHasher>
where
    T: Copy,
    NativeValue<T>: Eq + Hash,
{
    let mut counts = HashMap::with_hasher(FxBuildHasher);
    for (&value, &valid) in values.iter().zip(validity) {
        if valid {
            *counts.entry(NativeValue(value)).or_insert(0) += 1;
        }
    }
    counts
}

/// Compare native counts, runs, partial exchange, and empty merge identities.
fn assert_integer_reference<T>(
    values: Vec<T>,
    valid: Vec<bool>,
    get_map: impl Fn(&IntegerFrequenciesPartial) -> &HashMap<NativeValue<T>, u32, FxBuildHasher>,
) -> VortexResult<()>
where
    T: IntegerPType,
    NativeValue<T>: Eq + Hash,
{
    let expected = reference_frequencies(&values, &valid);
    let mut expected_runs = 0u64;
    let mut previous = None;
    for (&value, &valid) in values.iter().zip(&valid) {
        if valid && previous.replace(value) != Some(value) {
            expected_runs += 1;
        }
    }
    let array = PrimitiveArray::new(Buffer::from(values), Validity::from_iter(valid));
    let dtypes =
        AggregateDTypes::try_new(&IntegerFrequencies, &EmptyOptions, array.dtype().clone())?;
    let args = dtypes.args(&EmptyOptions);
    let partial = summarize(&IntegerFrequencies, array)?;
    assert_eq!(get_map(&partial), &expected);
    assert_eq!(partial.run_summary().run_count(), expected_runs);
    assert_eq!(
        partial.run_summary().valid_count(),
        expected
            .values()
            .map(|&count| u64::from(count))
            .sum::<u64>()
    );
    assert_eq!(partial.distinct_count() as usize, expected.len());

    let scalar = IntegerFrequencies.to_scalar(args, &partial)?;
    let parsed = IntegerFrequencies.partial_from_scalar(args, scalar)?;
    assert_eq!(get_map(&parsed), &expected);
    assert_eq!(parsed.run_summary().run_count(), expected_runs);
    let empty = IntegerFrequencies.empty_partial(args)?;
    let merged = IntegerFrequencies.merge_partials(args, empty, partial.clone())?;
    assert_eq!(get_map(&merged), &expected);
    let empty = IntegerFrequencies.empty_partial(args)?;
    let merged = IntegerFrequencies.merge_partials(args, partial, empty)?;
    assert_eq!(get_map(&merged), &expected);
    assert_eq!(
        IntegerFrequencies.finalize_scalar(args, &merged)?,
        Scalar::from(expected.len() as u64),
    );
    Ok(())
}

#[rstest]
#[case::u8(PType::U8)]
#[case::u16(PType::U16)]
#[case::u32(PType::U32)]
#[case::u64(PType::U64)]
#[case::i8(PType::I8)]
#[case::i16(PType::I16)]
#[case::i32(PType::I32)]
#[case::i64(PType::I64)]
fn integer_counts_match_reference(
    #[case] ptype: PType,
    #[values(0, 1, 63, 64, 65, 1297)] len: usize,
    #[values(1, 3, 64, 100)] run_len: usize,
    #[values(None, Some(3), Some(97))] null_every: Option<usize>,
) -> VortexResult<()> {
    macro_rules! check {
        ($T:ty, $variant:ident) => {{
            let values = (0..len)
                .map(|index| {
                    <$T>::try_from((index / run_len) % 16)
                        .vortex_expect("reference values fit every native integer type")
                })
                .collect();
            let valid = (0..len)
                .map(|index| null_every.is_none_or(|every| index % every != 0))
                .collect();
            assert_integer_reference(values, valid, |partial| {
                let IntegerFrequenciesPartial::$variant(values) = partial else {
                    unreachable!()
                };
                values.frequencies()
            })
        }};
    }

    match ptype {
        PType::U8 => check!(u8, U8),
        PType::U16 => check!(u16, U16),
        PType::U32 => check!(u32, U32),
        PType::U64 => check!(u64, U64),
        PType::I8 => check!(i8, I8),
        PType::I16 => check!(i16, I16),
        PType::I32 => check!(i32, I32),
        PType::I64 => check!(i64, I64),
        _ => unreachable!(),
    }
}

#[test]
fn integer_extreme_values_and_null_payloads() -> VortexResult<()> {
    assert_integer_reference(
        vec![i64::MIN, i64::MAX, i64::MIN, 0, 0],
        vec![true, true, true, false, false],
        |partial| {
            let IntegerFrequenciesPartial::I64(values) = partial else {
                unreachable!()
            };
            values.frequencies()
        },
    )?;
    assert_integer_reference(vec![0u64, u64::MAX, 0], vec![true, true, true], |partial| {
        let IntegerFrequenciesPartial::U64(values) = partial else {
            unreachable!()
        };
        values.frequencies()
    })?;
    assert_integer_reference(
        vec![1000i32, 1001, 1000, 5_000_000, 1003],
        vec![true, true, true, false, true],
        |partial| {
            let IntegerFrequenciesPartial::I32(values) = partial else {
                unreachable!()
            };
            values.frequencies()
        },
    )
}

#[test]
fn integer_merge_adds_frequencies() -> VortexResult<()> {
    let first = PrimitiveArray::from_option_iter([Some(10i32), None, Some(11), Some(10)]);
    let second = PrimitiveArray::from_option_iter([None, Some(10i32), Some(12)]);
    let dtypes =
        AggregateDTypes::try_new(&IntegerFrequencies, &EmptyOptions, first.dtype().clone())?;
    let args = dtypes.args(&EmptyOptions);
    let first = summarize(&IntegerFrequencies, first)?;
    let second = summarize(&IntegerFrequencies, second)?;
    let merged = IntegerFrequencies.merge_partials(args, first, second)?;
    let IntegerFrequenciesPartial::I32(values) = &merged else {
        unreachable!()
    };
    let values = values.frequencies();
    assert_eq!(values.get(&NativeValue(10)), Some(&3));
    assert_eq!(values.get(&NativeValue(11)), Some(&1));
    assert_eq!(values.get(&NativeValue(12)), Some(&1));
    assert_eq!(
        merged.most_frequent_value_and_count(),
        Some((10i32.into(), 3))
    );
    assert_eq!(merged.run_summary().valid_count(), 5);
    assert_eq!(merged.run_summary().run_count(), 4);
    Ok(())
}

#[test]
fn float_distinct_preserves_bitwise_values() -> VortexResult<()> {
    let nan1 = f32::from_bits(0x7fc0_0001);
    let nan2 = f32::from_bits(0x7fc0_0002);
    let array = PrimitiveArray::new(
        buffer![0.0f32, -0.0, nan1, nan2, nan1, 99.0],
        Validity::from_iter([true, true, true, true, true, false]),
    );
    let dtypes = AggregateDTypes::try_new(&FloatDistinct, &EmptyOptions, array.dtype().clone())?;
    let args = dtypes.args(&EmptyOptions);
    let partial = summarize(&FloatDistinct, array)?;
    let FloatDistinctPartial::F32(values) = &partial else {
        unreachable!()
    };
    let values = values.values();
    assert_eq!(values.len(), 4);
    assert!(values.contains(&NativeValue(0.0)));
    assert!(values.contains(&NativeValue(-0.0)));
    assert!(values.contains(&NativeValue(nan1)));
    assert!(values.contains(&NativeValue(nan2)));

    let scalar = FloatDistinct.to_scalar(args, &partial)?;
    let parsed = FloatDistinct.partial_from_scalar(args, scalar)?;
    let FloatDistinctPartial::F32(parsed_values) = &parsed else {
        unreachable!()
    };
    assert_eq!(parsed_values.values(), values);
    assert_eq!(
        parsed.run_summary().run_count(),
        partial.run_summary().run_count()
    );
    assert_eq!(
        parsed.run_summary().valid_count(),
        partial.run_summary().valid_count()
    );
    assert_eq!(
        FloatDistinct.finalize_scalar(args, &parsed)?,
        Scalar::from(4u64)
    );
    let merged = FloatDistinct.merge_partials(args, partial, parsed)?;
    assert_eq!(merged.distinct_count(), 4);
    Ok(())
}

#[test]
fn float_half_nan_payloads_survive_partial_roundtrip() -> VortexResult<()> {
    let array = PrimitiveArray::new(
        buffer![
            f16::from_bits(0x7e01),
            f16::from_bits(0x7e02),
            f16::from_bits(0xfe01)
        ],
        Validity::NonNullable,
    );
    let dtypes = AggregateDTypes::try_new(&FloatDistinct, &EmptyOptions, array.dtype().clone())?;
    let args = dtypes.args(&EmptyOptions);
    let partial = summarize(&FloatDistinct, array)?;
    let scalar = FloatDistinct.to_scalar(args, &partial)?;
    let parsed = FloatDistinct.partial_from_scalar(args, scalar)?;
    let FloatDistinctPartial::F16(expected) = &partial else {
        unreachable!()
    };
    let FloatDistinctPartial::F16(actual) = &parsed else {
        unreachable!()
    };
    assert_eq!(actual.values(), expected.values());
    assert_eq!(
        parsed.run_summary().run_count(),
        partial.run_summary().run_count()
    );
    Ok(())
}

#[rstest]
#[case::empty(PrimitiveArray::from_iter(std::iter::empty::<f64>()))]
#[case::all_null(PrimitiveArray::from_option_iter::<f64, _>([None, None]))]
fn float_empty_identity(#[case] array: PrimitiveArray) -> VortexResult<()> {
    let dtypes = AggregateDTypes::try_new(&FloatDistinct, &EmptyOptions, array.dtype().clone())?;
    let args = dtypes.args(&EmptyOptions);
    let partial = summarize(&FloatDistinct, array)?;
    assert_eq!(partial.distinct_count(), 0);
    let scalar = FloatDistinct.to_scalar(args, &partial)?;
    let parsed = FloatDistinct.partial_from_scalar(args, scalar)?;
    let merged = FloatDistinct.merge_partials(args, partial, parsed)?;
    assert_eq!(
        FloatDistinct.finalize_scalar(args, &merged)?,
        Scalar::from(0u64)
    );
    Ok(())
}

#[test]
fn float_run_merge_skips_all_null_chunks() -> VortexResult<()> {
    let nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let first = PrimitiveArray::from_option_iter([None, Some(nan), Some(nan), Some(0.0f64)]);
    let empty = PrimitiveArray::from_option_iter::<f64, _>([None, None]);
    let last = PrimitiveArray::from_option_iter([None, Some(-0.0f64), Some(5.0)]);
    let whole = PrimitiveArray::from_option_iter([
        None,
        Some(nan),
        Some(nan),
        Some(0.0f64),
        None,
        None,
        None,
        Some(-0.0),
        Some(5.0),
    ]);
    let dtypes = AggregateDTypes::try_new(&FloatDistinct, &EmptyOptions, first.dtype().clone())?;
    let args = dtypes.args(&EmptyOptions);
    let first = summarize(&FloatDistinct, first)?;
    let empty = summarize(&FloatDistinct, empty)?;
    let last = summarize(&FloatDistinct, last)?;
    let whole = summarize(&FloatDistinct, whole)?;
    let merged = FloatDistinct.merge_partials(args, first, empty)?;
    let merged = FloatDistinct.merge_partials(args, merged, last)?;
    assert_eq!(merged.distinct_count(), whole.distinct_count());
    assert_eq!(merged.run_summary().valid_count(), 5);
    assert_eq!(merged.run_summary().run_count(), 4);
    assert!(merged.run_summary().first_is_nan());
    assert_eq!(
        merged.run_summary().average_run_length_for_compression()?,
        1
    );
    assert_eq!(
        merged.run_summary().average_run_length_for_compression()?,
        whole.run_summary().average_run_length_for_compression()?,
    );
    Ok(())
}
