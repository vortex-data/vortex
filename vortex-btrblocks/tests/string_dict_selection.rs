// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regression tests for bounded string dictionary selection.

#![cfg(test)]
#![allow(clippy::tests_outside_test_module)]

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Dict;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::schemes::string::FSSTScheme;
use vortex_btrblocks::schemes::string::StringDictScheme;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[derive(Clone, Copy, Debug)]
enum Distribution {
    Uniform,
    Clustered,
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
    let compressed = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&StringDictScheme)
        .with_new_scheme(&FSSTScheme)
        .build()
        .compress(&array, &mut ctx)?;

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
