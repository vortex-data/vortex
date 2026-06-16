// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests to verify that each string compression scheme produces the expected encoding.

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

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

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_onpair_in_default_scheme_list() {
    use crate::SchemeExt;
    use crate::schemes::string::OnPairScheme;

    let ids: Vec<_> = crate::ALL_SCHEMES.iter().map(|s| s.id()).collect();
    assert!(
        ids.contains(&OnPairScheme.id()),
        "OnPairScheme not registered in ALL_SCHEMES"
    );
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn test_onpair_compressed() -> VortexResult<()> {
    // Dictionary-style string corpus: high lexical overlap, short rows.
    // OnPair beats FSST on this corpus, so it wins the sample-based
    // comparison even though both are registered by default.
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

/// FSST is registered in the default scheme list (alongside OnPair), and an
/// FSST-only builder still produces an FSST array.
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
