// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct VarBinFixture;

impl FlatLayoutFixture for VarBinFixture {
    fn name(&self) -> &str {
        "varbin.vortex"
    }

    fn description(&self) -> &str {
        "VarBin-encoded strings including empty, unicode, emoji, and a nullable column"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![VarBin::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let strings = VarBinArray::from(vec!["", "hello", "こんにちは", "\u{1f980}"]);
        let nullable_strings =
            VarBinArray::from(vec![Some("hello"), None, Some("world"), Some("")]);
        let arr = StructArray::try_new(
            FieldNames::from(["text", "nullable_text"]),
            vec![strings.into_array(), nullable_strings.into_array()],
            4,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
