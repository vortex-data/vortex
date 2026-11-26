// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::Admission;
use super::HEAD_PROBE_RANGE_SIZE;
use super::InputProfile;
use super::MAX_DICTIONARY_ENTRIES;
use super::MAX_PROBE_BYTES;
use super::PROBE_RANGE_SIZE;
use super::ProbeSample;
use super::admit_trial;
use super::build_candidate;
use super::bytes_dictionary_estimate;
use crate::CascadingCompressor;
use crate::builtins::StringDictScheme;
use crate::compressor::TrialOutput;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::EstimateScore;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// Builds compression data for candidate tests.
fn string_data(values: &[Option<String>]) -> ArrayAndStats {
    let nullability = if values.iter().any(Option::is_none) {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let array = VarBinViewArray::from_iter(
        values.iter().map(|value| value.as_deref()),
        DType::Utf8(nullability),
    )
    .into_array();
    ArrayAndStats::new(array, Default::default())
}

/// Returns `rows` values that cycle through `distinct` outlined strings.
fn cycled(rows: usize, distinct: usize) -> Vec<Option<String>> {
    (0..rows)
        .map(|index| {
            let value = index % distinct;
            Some(format!(
                "{:016x}-value-{value:08x}",
                value.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ))
        })
        .collect()
}

/// Profiles and probes the input.
fn profile_and_probe(
    data: &ArrayAndStats,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<(InputProfile, Option<ProbeSample>)> {
    let profile = InputProfile::new(data, exec_ctx)?;
    let sample = ProbeSample::new(
        &data.array_as_varbinview(),
        &profile.validity,
        PROBE_RANGE_SIZE,
    );
    Ok((profile, sample))
}

/// Resolves the Dictionary estimate against an optional threshold.
fn estimate_dictionary(
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    best_so_far: Option<EstimateScore>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<EstimateVerdict> {
    let callback = match bytes_dictionary_estimate(&StringDictScheme) {
        CompressionEstimate::Deferred(DeferredEstimate::Callback(callback)) => callback,
        estimate => return Err(vortex_err!("dictionary estimation returned {estimate:?}")),
    };
    callback(
        compressor,
        data,
        best_so_far,
        CompressorContext::new(),
        exec_ctx,
    )
}

#[test]
fn complete_probe_counts_distinct_exactly() -> VortexResult<()> {
    let data = string_data(&cycled(4096, 100));
    let mut exec_ctx = SESSION.create_execution_ctx();
    let (profile, sample) = profile_and_probe(&data, &mut exec_ctx)?;
    let sample = sample.ok_or_else(|| vortex_err!("probe did not finish"))?;

    assert!(sample.complete);
    assert_eq!(sample.estimated_distinct(profile.valid_rows), 100.0);
    Ok(())
}

#[rstest]
#[case(20_000)]
#[case(50_000)]
fn partial_probe_estimates_distinct_beyond_probe_size(#[case] distinct: usize) -> VortexResult<()> {
    let data = string_data(&cycled(1 << 20, distinct));
    let mut exec_ctx = SESSION.create_execution_ctx();
    let (profile, sample) = profile_and_probe(&data, &mut exec_ctx)?;
    let sample = sample.ok_or_else(|| vortex_err!("probe did not finish"))?;
    let estimate = sample.estimated_distinct(profile.valid_rows);

    assert!(!sample.complete);
    // The probe sees at most 8192 rows, so its raw distinct count cannot reach `distinct`.
    assert!(sample.distinct < distinct);
    let error = (estimate - distinct as f64).abs() / distinct as f64;
    assert!(error < 0.25, "estimated {estimate} for {distinct}");
    Ok(())
}

#[test]
fn probe_is_inconclusive_above_byte_budget() -> VortexResult<()> {
    let data = string_data(&[Some("a".to_owned()), Some("x".repeat(MAX_PROBE_BYTES))]);
    let mut exec_ctx = SESSION.create_execution_ctx();
    let (_, sample) = profile_and_probe(&data, &mut exec_ctx)?;

    assert_eq!(sample, None);
    Ok(())
}

#[test]
fn nulls_count_as_one_dictionary_entry() -> VortexResult<()> {
    let values = (0..8192)
        .map(|index| (index % 10 == 0).then(|| format!("unique-non-null-value-{index:08x}")))
        .collect::<Vec<_>>();
    let data = string_data(&values);
    let mut exec_ctx = SESSION.create_execution_ctx();
    let (profile, sample) = profile_and_probe(&data, &mut exec_ctx)?;
    let sample = sample.ok_or_else(|| vortex_err!("probe did not finish"))?;

    assert_eq!(profile.valid_rows, 820);
    assert_eq!(sample.rows, 820);
    assert_eq!(admit_trial(&profile, &sample, None), Admission::Trial);
    Ok(())
}

#[test]
fn unique_values_are_not_admitted() -> VortexResult<()> {
    let data = string_data(&cycled(8192, 8192));
    let compressor = CascadingCompressor::new(vec![&StringDictScheme]);
    let mut exec_ctx = SESSION.create_execution_ctx();
    let (profile, sample) = profile_and_probe(&data, &mut exec_ctx)?;
    let sample = sample.ok_or_else(|| vortex_err!("probe did not finish"))?;

    assert_eq!(admit_trial(&profile, &sample, None), Admission::Reject);
    assert!(matches!(
        estimate_dictionary(&compressor, &data, None, &mut exec_ctx)?,
        EstimateVerdict::Skip
    ));
    Ok(())
}

#[test]
fn threshold_rejects_trial_before_building() -> VortexResult<()> {
    let data = string_data(&cycled(65_536, 4096));
    let mut exec_ctx = SESSION.create_execution_ctx();
    let (profile, sample) = profile_and_probe(&data, &mut exec_ctx)?;
    let sample = sample.ok_or_else(|| vortex_err!("probe did not finish"))?;

    assert_eq!(admit_trial(&profile, &sample, None), Admission::Trial);
    assert_eq!(
        admit_trial(
            &profile,
            &sample,
            Some(EstimateScore::FiniteCompression(1000.0))
        ),
        Admission::Reject
    );
    Ok(())
}

#[test]
fn too_many_estimated_entries_defer_to_sampling() -> VortexResult<()> {
    // Each value repeats 16 times, so a dictionary wins even though it exceeds the trial bounds.
    let distinct = 2 * MAX_DICTIONARY_ENTRIES;
    let values = (0..16 * distinct)
        .map(|index| Some(format!("{:016x}", (index * 7919) % distinct)))
        .collect::<Vec<_>>();
    let data = string_data(&values);
    let mut exec_ctx = SESSION.create_execution_ctx();
    let (profile, sample) = profile_and_probe(&data, &mut exec_ctx)?;
    let sample = sample.ok_or_else(|| vortex_err!("probe did not finish"))?;

    assert_eq!(admit_trial(&profile, &sample, None), Admission::TooLarge);
    Ok(())
}

#[test]
fn candidate_stops_when_dictionary_storage_exceeds_limit() -> VortexResult<()> {
    let suffix = "x".repeat(70_000);
    let values = (0..128)
        .map(|index| Some(format!("{:08x}-{suffix}", index % 64)))
        .collect::<Vec<_>>();
    let data = string_data(&values);
    let mut exec_ctx = SESSION.create_execution_ctx();

    assert!(build_candidate(data.array(), &mut exec_ctx)?.is_none());
    Ok(())
}

#[test]
fn candidate_uses_full_u16_code_range() -> VortexResult<()> {
    let mut values = (0..=u16::MAX)
        .map(|value| Some(format!("{value:08x}")))
        .collect::<Vec<_>>();
    let mut exec_ctx = SESSION.create_execution_ctx();
    assert!(build_candidate(string_data(&values).array(), &mut exec_ctx)?.is_some());

    values.push(Some("past-u16-code-range".to_owned()));
    assert!(build_candidate(string_data(&values).array(), &mut exec_ctx)?.is_none());
    Ok(())
}

#[test]
fn completed_candidate_is_reused() -> VortexResult<()> {
    let data = string_data(&cycled(65_536, 4096));
    let compressor = CascadingCompressor::new(vec![&StringDictScheme]);
    let mut exec_ctx = SESSION.create_execution_ctx();
    let verdict = estimate_dictionary(&compressor, &data, None, &mut exec_ctx)?;
    assert!(matches!(verdict, EstimateVerdict::Ratio(_)));
    let cached = data
        .get::<TrialOutput>()
        .ok_or_else(|| vortex_err!("completed candidate was not cached"))?;
    let compressed =
        StringDictScheme.compress(&compressor, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(ArrayRef::ptr_eq(&cached.array, &compressed));
    Ok(())
}

/// Repeats far apart still reach the head probe: each value below recurs half the input away.
#[rstest]
#[case::unique(65_536, 65_536, true)]
#[case::two_repeats_per_value(65_536, 32_768, false)]
#[case::four_repeats_per_value(65_536, 16_384, false)]
#[case::two_repeats_in_a_shorter_input(16_384, 8_192, false)]
fn head_probe_screens_unique_inputs(
    #[case] rows: usize,
    #[case] distinct: usize,
    #[case] expected: bool,
) -> VortexResult<()> {
    let data = string_data(&cycled(rows, distinct));
    let mut exec_ctx = SESSION.create_execution_ctx();
    let profile = InputProfile::new(&data, &mut exec_ctx)?;
    let head = ProbeSample::new(
        &data.array_as_varbinview(),
        &profile.validity,
        HEAD_PROBE_RANGE_SIZE,
    )
    .ok_or_else(|| vortex_err!("probe did not finish"))?;

    assert!(!head.complete);
    assert_eq!(head.is_unique(), expected);
    if expected {
        assert_eq!(admit_trial(&profile, &head, None), Admission::Reject);
    }
    Ok(())
}

/// A mostly-null column of distinct long values compresses far better as sparse values than as a
/// dictionary, whose distinct values no string scheme compresses more than eight-fold.
#[test]
fn values_ratio_bound_rejects_sparse_unique_values() -> VortexResult<()> {
    let suffix = "x".repeat(240);
    let values = (0..40_960)
        .map(|index| (index % 23 == 0).then(|| format!("{index:016x}{suffix}")))
        .collect::<Vec<_>>();
    let data = string_data(&values);
    let mut exec_ctx = SESSION.create_execution_ctx();
    let (profile, sample) = profile_and_probe(&data, &mut exec_ctx)?;
    let sample = sample.ok_or_else(|| vortex_err!("probe did not finish"))?;

    assert_eq!(admit_trial(&profile, &sample, None), Admission::Trial);
    assert_eq!(
        admit_trial(
            &profile,
            &sample,
            Some(EstimateScore::FiniteCompression(22.0))
        ),
        Admission::Reject
    );
    Ok(())
}
