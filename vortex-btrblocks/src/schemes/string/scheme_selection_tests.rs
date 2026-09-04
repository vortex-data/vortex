// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests to verify that each string compression scheme produces the expected encoding.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorBuilder;
use crate::schemes::string::FSSTScheme;
use crate::schemes::string::StringDictScheme;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[derive(Clone, Copy, Debug)]
enum Distribution {
    Uniform,
    Clustered,
}

#[test]
fn test_constant_compressed() -> VortexResult<()> {
    let strings: Vec<Option<&str>> = vec![Some("constant_value"); 100];
    let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
    let array_ref = array.into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn test_dict_compressed() -> VortexResult<()> {
    let distinct_values = ["apple", "banana", "cherry"];
    let mut strings = Vec::with_capacity(1000);
    for i in 0..1000 {
        strings.push(Some(distinct_values[i % 3]));
    }
    let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
    let array_ref = array.into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Dict>());
    Ok(())
}

#[rstest]
#[case::outlined_utf8(4096, 28, false, Distribution::Uniform)]
#[case::nullable_utf8(4096, 28, true, Distribution::Uniform)]
#[case::outlined_utf8_8192(8192, 28, false, Distribution::Uniform)]
#[case::clustered_utf8(4096, 28, false, Distribution::Clustered)]
fn test_dict_compressed_with_more_values_than_sample(
    #[case] distinct_count: usize,
    #[case] value_length: usize,
    #[case] nullable: bool,
    #[case] distribution: Distribution,
) -> VortexResult<()> {
    let distinct_values = (0..distinct_count)
        .map(|value| {
            let mut string = format!("common-prefix-value-{value:08x}");
            string.extend(std::iter::repeat_n(
                'x',
                value_length.saturating_sub(string.len()),
            ));
            string
        })
        .collect::<Vec<_>>();
    let values = (0..65_536)
        .map(|index| {
            let value_index = match distribution {
                Distribution::Uniform => index % distinct_values.len(),
                Distribution::Clustered => index * distinct_values.len() / 65_536,
            };
            (!nullable || index % 10 != 0).then_some(distinct_values[value_index].as_bytes())
        })
        .collect::<Vec<_>>();
    let nullability = if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let array = VarBinViewArray::from_iter(values, DType::Utf8(nullability)).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&StringDictScheme)
        .with_new_scheme(&FSSTScheme)
        .build();
    let compressed = compressor.compress(&array, &mut ctx)?;

    assert!(
        compressed.is::<Dict>(),
        "expected Dict, got {}",
        compressed.encoding_id()
    );
    assert_arrays_eq!(&array, &compressed, &mut ctx);
    Ok(())
}

#[test]
fn test_unique_strings_with_common_prefix_not_dict_compressed() -> VortexResult<()> {
    let values = (0usize..4096)
        .map(|value| Some(format!("common-prefix-value-{value:08x}")))
        .collect::<Vec<_>>();
    let array =
        VarBinViewArray::from_iter(values, DType::Utf8(Nullability::NonNullable)).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = BtrBlocksCompressor::default().compress(&array, &mut ctx)?;

    assert!(
        !compressed.is::<Dict>(),
        "expected a non-Dict encoding, got {}",
        compressed.encoding_id()
    );
    assert_arrays_eq!(&array, &compressed, &mut ctx);
    Ok(())
}

#[test]
fn test_sample_fallback_can_select_dict() -> VortexResult<()> {
    let suffix = "x".repeat((1 << 20) + 1);
    let distinct_values = [format!("first-{suffix}"), format!("second-{suffix}")];
    let values = (0..4)
        .map(|index| distinct_values[index % distinct_values.len()].as_str())
        .collect::<Vec<_>>();
    let array = VarBinViewArray::from_iter_str(values).into_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&StringDictScheme)
        .with_new_scheme(&FSSTScheme)
        .build();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&array, &mut ctx)?;

    assert!(
        compressed.is::<Dict>(),
        "expected Dict, got {}",
        compressed.encoding_id()
    );
    assert_arrays_eq!(&array, &compressed, &mut ctx);
    Ok(())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_unstable_all_schemes_includes_onpair() {
    use crate::SchemeExt;
    use crate::schemes::string::onpair::OnPairScheme;

    let ids: Vec<_> = crate::ALL_SCHEMES.iter().map(|s| s.id()).collect();
    assert!(
        ids.contains(&OnPairScheme.id()),
        "OnPairScheme not registered in ALL_SCHEMES"
    );
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_unstable_default_btrblocks_compressor_selects_onpair() -> VortexResult<()> {
    // Dictionary-style string corpus: high lexical overlap, short rows.
    // OnPair beats FSST on this corpus, so it wins the sample-based
    // comparison even though both are registered when `unstable_encodings`
    // is enabled.
    let mut strings = Vec::with_capacity(1000);
    for i in 0..1000 {
        strings.push(Some(format!(
            "this_is_a_common_prefix_with_some_variation_{i}_and_a_common_suffix_pattern"
        )));
    }
    let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
    let array_ref = array.into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert!(
        compressed.is::<vortex_onpair::OnPair>(),
        "expected OnPair, got {}",
        compressed.encoding_id()
    );
    Ok(())
}

/// FSST is registered in the default scheme list, and an FSST-only builder
/// still produces an FSST array.
#[test]
fn test_fsst_in_default_scheme_list() -> VortexResult<()> {
    use crate::BtrBlocksCompressorBuilder;
    use crate::SchemeExt;
    use crate::schemes::string::FSSTScheme;

    // FSST is registered by default.
    assert!(
        crate::ALL_SCHEMES.iter().any(|s| s.id() == FSSTScheme.id()),
        "FSSTScheme should be in ALL_SCHEMES",
    );

    // An FSST-only builder still produces an FSST array for FSST-favourable
    // input.
    let mut strings = Vec::with_capacity(1000);
    for i in 0..1000 {
        strings.push(Some(format!(
            "this_is_a_common_prefix_with_some_variation_{i}_and_a_common_suffix_pattern"
        )));
    }
    let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
    let array_ref = array.into_array();

    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&FSSTScheme)
        .build();
    let compressed = compressor.compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert!(
        compressed.is::<FSST>(),
        "expected FSST when only FSSTScheme is registered, got {}",
        compressed.encoding_id()
    );
    Ok(())
}
