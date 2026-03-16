// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
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
        vec![FSST::ID]
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
        let url_col = VarBinArray::from(url_refs);

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
        let log_col = VarBinArray::from(log_refs);

        let nullable_urls: Vec<Option<String>> = (0..N)
            .map(|i| (i % 7 != 0).then(|| format!("{}{}", prefixes[i % prefixes.len()], i * 3)))
            .collect();
        let nullable_refs: Vec<Option<&str>> = nullable_urls.iter().map(|s| s.as_deref()).collect();
        let nullable_col = VarBinArray::from(nullable_refs);

        let short_tokens = ["a", "bb", "ccc", "dd", "e"];
        let short_strs: Vec<&str> = (0..N)
            .map(|i| short_tokens[i % short_tokens.len()])
            .collect();
        let short_col = VarBinArray::from(short_strs);

        let url_comp = fsst_train_compressor(&url_col);
        let log_comp = fsst_train_compressor(&log_col);
        let nullable_comp = fsst_train_compressor(&nullable_col);
        let short_comp = fsst_train_compressor(&short_col);

        let arr = StructArray::try_new(
            FieldNames::from(["urls", "logs", "nullable_urls", "short_strs"]),
            vec![
                fsst_compress(url_col, &url_comp).into_array(),
                fsst_compress(log_col, &log_comp).into_array(),
                fsst_compress(nullable_col, &nullable_comp).into_array(),
                fsst_compress(short_col, &short_comp).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
