// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct VarBinViewFixture;

impl FlatLayoutFixture for VarBinViewFixture {
    fn name(&self) -> &str {
        "varbinview.vortex"
    }

    fn description(&self) -> &str {
        "VarBinView-encoded strings including empty, unicode, emoji, long (>12 byte) strings, and a nullable column"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![VarBinView::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let strings = VarBinViewArray::from_iter_bin(vec!["", "hello", "こんにちは", "\u{1f980}"]);
        let nullable_strings = VarBinViewArray::from_iter_nullable_str(vec![
            Some("hello"),
            None,
            Some("world"),
            Some(""),
        ]);
        // Strings >12 bytes exercise VarBinView's buffer-reference mechanism (out-of-line storage).
        let long_strings = VarBinViewArray::from_iter_str(vec![
            "short",
            "this string is definitely longer than twelve bytes",
            "another-long-string-that-exceeds-inline-limit",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789",
        ]);
        let arr = StructArray::try_new(
            FieldNames::from(["text", "nullable_text", "long_text"]),
            vec![
                strings.into_array(),
                nullable_strings.into_array(),
                long_strings.into_array(),
            ],
            4,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
