// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::fsst::FSST;
use vortex::encodings::fsst::fsst_compress;
use vortex::encodings::fsst::fsst_train_compressor;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct FsstFixture;

impl FlatLayoutFixture for FsstFixture {
    fn name(&self) -> &str {
        "fsst.vortex"
    }

    fn description(&self) -> &str {
        "Strings with common substrings/prefixes for FSST encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![FSST.id()]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let prefixes = [
            "https://example.com/api/v1/users/",
            "https://example.com/api/v1/orders/",
            "https://example.com/api/v1/products/",
            "https://example.com/api/v2/users/",
        ];
        let urls: Vec<String> = (0..N)
            .map(|i| format!("{}{}", prefixes[i % prefixes.len()], i))
            .collect();
        let url_refs: Vec<&str> = urls.iter().map(|s| s.as_str()).collect();
        let url_col = VarBinArray::from_strs(url_refs);

        let severities = ["INFO", "WARN", "ERROR", "DEBUG"];
        let components = ["auth", "db", "cache", "api"];
        let logs: Vec<String> = (0..N)
            .map(|i| {
                format!(
                    "[{}] {}: request processed in {}ms",
                    severities[i % severities.len()],
                    components[i % components.len()],
                    i % 1000
                )
            })
            .collect();
        let log_refs: Vec<&str> = logs.iter().map(|s| s.as_str()).collect();
        let log_col = VarBinArray::from_strs(log_refs);

        let nullable_urls: Vec<Option<String>> = (0..N)
            .map(|i| (i % 7 != 0).then(|| format!("{}{}", prefixes[i % prefixes.len()], i * 3)))
            .collect();
        let nullable_refs: Vec<Option<&str>> = nullable_urls.iter().map(|s| s.as_deref()).collect();
        let nullable_col = VarBinArray::from_nullable_strs(nullable_refs);

        let short_tokens = ["a", "bb", "ccc", "dd", "e"];
        let short_strs: Vec<&str> = (0..N)
            .map(|i| short_tokens[i % short_tokens.len()])
            .collect();
        let short_col = VarBinArray::from_strs(short_strs);
        let empty_and_unicode_values =
            ["", "こんにちは", "😀", "naive", "façade", "résumé", "مرحبا"];
        let empty_and_unicode: Vec<&str> = (0..N)
            .map(|i| empty_and_unicode_values[i % empty_and_unicode_values.len()])
            .collect();
        let empty_and_unicode_col = VarBinArray::from_strs(empty_and_unicode);
        let suffix_shared_values: Vec<String> = (0..N)
            .map(|i| format!("prefix-{:04}-common-suffix", i % 64))
            .collect();
        let suffix_shared_refs: Vec<&str> =
            suffix_shared_values.iter().map(String::as_str).collect();
        let suffix_shared_col = VarBinArray::from_strs(suffix_shared_refs);
        let high_entropy_values: Vec<String> = (0..N)
            .map(|i| format!("{:016x}{:016x}", i.wrapping_mul(97), i.wrapping_mul(13_579)))
            .collect();
        let high_entropy_refs: Vec<&str> = high_entropy_values.iter().map(String::as_str).collect();
        let high_entropy_col = VarBinArray::from_strs(high_entropy_refs);
        let all_null_clustered = VarBinArray::from_nullable_strs(
            (0..N)
                .map(|i| {
                    if !(16..N - 16).contains(&i) {
                        None
                    } else {
                        Some("clustered-null-middle")
                    }
                })
                .collect::<Vec<_>>(),
        );

        let url_comp = fsst_train_compressor(&url_col);
        let log_comp = fsst_train_compressor(&log_col);
        let nullable_comp = fsst_train_compressor(&nullable_col);
        let short_comp = fsst_train_compressor(&short_col);
        let empty_and_unicode_comp = fsst_train_compressor(&empty_and_unicode_col);
        let suffix_shared_comp = fsst_train_compressor(&suffix_shared_col);
        let high_entropy_comp = fsst_train_compressor(&high_entropy_col);
        let all_null_clustered_comp = fsst_train_compressor(&all_null_clustered);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arr = StructArray::try_new(
            FieldNames::from([
                "urls",
                "logs",
                "nullable_urls",
                "short_strs",
                "empty_and_unicode",
                "suffix_shared",
                "high_entropy",
                "all_null_clustered",
            ]),
            vec![
                fsst_compress(
                    &url_col,
                    url_col.len(),
                    url_col.dtype(),
                    &url_comp,
                    &mut ctx,
                )
                .into_array(),
                fsst_compress(
                    &log_col,
                    log_col.len(),
                    log_col.dtype(),
                    &log_comp,
                    &mut ctx,
                )
                .into_array(),
                fsst_compress(
                    &nullable_col,
                    nullable_col.len(),
                    nullable_col.dtype(),
                    &nullable_comp,
                    &mut ctx,
                )
                .into_array(),
                fsst_compress(
                    &short_col,
                    short_col.len(),
                    short_col.dtype(),
                    &short_comp,
                    &mut ctx,
                )
                .into_array(),
                fsst_compress(
                    &empty_and_unicode_col,
                    empty_and_unicode_col.len(),
                    empty_and_unicode_col.dtype(),
                    &empty_and_unicode_comp,
                    &mut ctx,
                )
                .into_array(),
                fsst_compress(
                    &suffix_shared_col,
                    suffix_shared_col.len(),
                    suffix_shared_col.dtype(),
                    &suffix_shared_comp,
                    &mut ctx,
                )
                .into_array(),
                fsst_compress(
                    &high_entropy_col,
                    high_entropy_col.len(),
                    high_entropy_col.dtype(),
                    &high_entropy_comp,
                    &mut ctx,
                )
                .into_array(),
                fsst_compress(
                    &all_null_clustered,
                    all_null_clustered.len(),
                    all_null_clustered.dtype(),
                    &all_null_clustered_comp,
                    &mut ctx,
                )
                .into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
