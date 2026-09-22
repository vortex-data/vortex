// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use onpair::search::ContainsScan;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::DEFAULT_CONFIG;
use crate::OnPair;
use crate::OnPairArray;
use crate::build_token_frequency_index;
use crate::onpair_compress;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

fn encode(values: &[Option<&str>], indexed: bool) -> VortexResult<OnPairArray> {
    let input = VarBinArray::from_iter(values.iter().copied(), DType::Utf8(Nullability::Nullable))
        .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = onpair_compress(&input, DEFAULT_CONFIG, &mut ctx)?
        .try_downcast::<OnPair>()
        .map_err(|array| vortex_err!("expected OnPair, got {}", array.encoding_id()))?;
    if indexed {
        build_token_frequency_index(encoded, &mut ctx)
    } else {
        Ok(encoded)
    }
}

fn execute_kernel(
    array: &OnPairArray,
    pattern: &str,
    options: LikeOptions,
) -> VortexResult<Option<ArrayRef>> {
    let pattern = ConstantArray::new(pattern, array.len()).into_array();
    <OnPair as LikeKernel>::like(
        array.as_view(),
        &pattern,
        options,
        &mut SESSION.create_execution_ctx(),
    )
}

#[rstest]
#[case("alpha", [Some(true), None, Some(false), Some(false), Some(false)])]
#[case("alpha%", [Some(true), None, Some(true), Some(false), Some(false)])]
fn supported_patterns(
    #[case] pattern: &str,
    #[case] expected: [Option<bool>; 5],
) -> VortexResult<()> {
    let array = encode(
        &[
            Some("alpha"),
            None,
            Some("alphabet"),
            Some("say hello"),
            Some(""),
        ],
        true,
    )?;
    let result = execute_kernel(&array, pattern, LikeOptions::default())?
        .ok_or_else(|| vortex_err!("OnPair should handle {pattern}"))?;
    assert_arrays_eq!(
        result,
        BoolArray::from_iter(expected),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn indexed_contains_handles_dense_matches() -> VortexResult<()> {
    let array = encode(
        &[
            Some("alpha"),
            None,
            Some("alphabet"),
            Some("say hello"),
            Some(""),
        ],
        true,
    )?;
    let result = execute_kernel(&array, "%ha%", LikeOptions::default())?
        .ok_or_else(|| vortex_err!("dense matches should use the compressed scan"))?;
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([Some(true), None, Some(true), Some(false), Some(false)]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn indexed_prefilter_supports_long_patterns() -> VortexResult<()> {
    let needle = "x".repeat(usize::from(u8::MAX) + 1);
    let matching = format!("before{needle}after");
    let pattern = format!("%{needle}%");

    let mut values = vec![Some("no match"); 4096];
    values[2048] = Some(&matching);
    let indexed = encode(&values, true)?;
    let result = execute_kernel(&indexed, &pattern, LikeOptions::default())?
        .ok_or_else(|| vortex_err!("indexed contains should use the compressed scan"))?;
    let mut expected = vec![Some(false); values.len()];
    expected[2048] = Some(true);
    assert_arrays_eq!(
        result,
        BoolArray::from_iter(expected),
        &mut SESSION.create_execution_ctx()
    );

    let unindexed = encode(&values, false)?;
    assert!(
        execute_kernel(&unindexed, &pattern, LikeOptions::default())?.is_none(),
        "unindexed contains should fall back to canonical LIKE"
    );
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn indexed_prefilter_handles_slices_and_negation() -> VortexResult<()> {
    let mut values = vec![Some("ordinary filler"); 1026];
    values[0] = Some("outside");
    values[100] = Some("rare hello needle");
    values[101] = None;
    values[1025] = Some("outside");
    let array = encode(&values, true)?
        .into_array()
        .slice(1..1025)?
        .try_downcast::<OnPair>()
        .map_err(|array| vortex_err!("expected sliced OnPair, got {}", array.encoding_id()))?;
    let result = execute_kernel(
        &array,
        "%hello%",
        LikeOptions {
            negated: true,
            case_insensitive: false,
        },
    )?
    .ok_or_else(|| vortex_err!("OnPair should handle indexed contains on a slice"))?;
    let expected = (1..1025).map(|index| values[index].map(|value| !value.contains("hello")));
    assert_arrays_eq!(
        result,
        BoolArray::from_iter(expected),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[rstest]
#[case("%hello%", "hello")]
#[case("%héllo%", "héllo")]
#[case("%a\\%b%", "a%b")]
#[case("%a\\_b%", "a_b")]
#[case("%a\\\\b%", "a\\b")]
#[case("%a long rare substring%", "a long rare substring")]
fn indexed_contains_returns_exact_matches(
    #[case] pattern: &str,
    #[case] needle: &str,
    #[values(false, true)] negated: bool,
) -> VortexResult<()> {
    let matching = format!("before{needle}after");
    let repeated = format!("{needle}::{needle}");
    let reversed: String = needle.chars().rev().collect();
    let values = [
        Some(matching.as_str()),
        Some(repeated.as_str()),
        Some(reversed.as_str()),
        None,
        Some(""),
        Some(needle),
        Some("ordinary filler"),
    ];
    let array = encode(&values, true)?;
    let result = execute_kernel(
        &array,
        pattern,
        LikeOptions {
            negated,
            case_insensitive: false,
        },
    )?
    .ok_or_else(|| vortex_err!("indexed contains should use the compressed scan"))?;
    let expected = values
        .iter()
        .map(|value| value.map(|value| value.contains(needle) != negated));
    assert_arrays_eq!(
        result,
        BoolArray::from_iter(expected),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[rstest]
fn empty_contains_preserves_nulls(
    #[values(false, true)] indexed: bool,
    #[values(false, true)] negated: bool,
) -> VortexResult<()> {
    let array = encode(&[Some(""), None, Some("alpha")], indexed)?;
    let result = execute_kernel(
        &array,
        "%%",
        LikeOptions {
            negated,
            case_insensitive: false,
        },
    )?
    .ok_or_else(|| vortex_err!("empty contains should not need a scan"))?;
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([Some(!negated), None, Some(!negated)]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn overlong_contains_falls_back() -> VortexResult<()> {
    let needle = "x".repeat(ContainsScan::MAX_PATTERN_LEN + 1);
    let pattern = format!("%{needle}%");
    let array = encode(&[Some(&needle), Some("no match"), None], true)?;
    assert!(execute_kernel(&array, &pattern, LikeOptions::default())?.is_none());

    let pattern = ConstantArray::new(pattern, array.len()).into_array();
    let result = Like::try_new(array.into_array(), pattern, LikeOptions::default())?
        .into_array()
        .execute::<BoolArray>(&mut SESSION.create_execution_ctx())?;
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([Some(true), Some(false), None]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[rstest]
#[case("%suffix", LikeOptions::default())]
#[case("a_b", LikeOptions::default())]
#[case("a%b", LikeOptions::default())]
#[case(
    "%alpha%",
    LikeOptions {
        negated: false,
        case_insensitive: true,
    }
)]
fn unsupported_patterns_fall_back(
    #[case] pattern: &str,
    #[case] options: LikeOptions,
) -> VortexResult<()> {
    let array = encode(&[Some("alpha")], true)?;
    assert!(execute_kernel(&array, pattern, options)?.is_none());
    Ok(())
}
