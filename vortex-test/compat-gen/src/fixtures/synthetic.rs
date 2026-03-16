// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::Bool;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_array::vtable::ArrayId;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::ArrayFixture;

struct PrimitivesFixture;

impl ArrayFixture for PrimitivesFixture {
    fn name(&self) -> &str {
        "primitives.vortex"
    }

    fn description(&self) -> &str {
        "All primitive types (u8-u64, i32, i64, f32, f64) including a nullable i32 column"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Primitive::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let arr = StructArray::try_new(
            FieldNames::from([
                "u8",
                "u16",
                "u32",
                "u64",
                "i32",
                "i64",
                "f32",
                "f64",
                "nullable_i32",
            ]),
            vec![
                PrimitiveArray::new(buffer![0u8, 128, 255], Validity::NonNullable).into_array(),
                PrimitiveArray::new(buffer![0u16, 32768, 65535], Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(
                    buffer![0u32, 2_147_483_648, 4_294_967_295],
                    Validity::NonNullable,
                )
                .into_array(),
                PrimitiveArray::new(
                    buffer![0u64, 9_223_372_036_854_775_808, u64::MAX],
                    Validity::NonNullable,
                )
                .into_array(),
                PrimitiveArray::new(buffer![i32::MIN, 0i32, i32::MAX], Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(buffer![i64::MIN, 0i64, i64::MAX], Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(buffer![f32::MIN, 0.0f32, f32::MAX], Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(buffer![f64::MIN, 0.0f64, f64::MAX], Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::from_option_iter([Some(1i32), None, Some(42)]).into_array(),
            ],
            3,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}

struct VarBinFixture;

impl ArrayFixture for VarBinFixture {
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

struct VarBinViewFixture;

impl ArrayFixture for VarBinViewFixture {
    fn name(&self) -> &str {
        "varbinview.vortex"
    }

    fn description(&self) -> &str {
        "VarBinView-encoded strings including empty, unicode, and emoji"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![VarBinView::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let strings = VarBinViewArray::from_iter_bin(vec!["", "hello", "こんにちは", "\u{1f980}"]);
        let arr = StructArray::try_new(
            FieldNames::from(["text"]),
            vec![strings.into_array()],
            4,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}

struct BooleansFixture;

impl ArrayFixture for BooleansFixture {
    fn name(&self) -> &str {
        "booleans.vortex"
    }

    fn description(&self) -> &str {
        "Boolean array with mixed true/false values"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Bool::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let bools = BoolArray::from_iter([true, false, true, true, false]);
        let arr = StructArray::try_new(
            FieldNames::from(["flag"]),
            vec![bools.into_array()],
            5,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}

struct StructNestedFixture;

impl ArrayFixture for StructNestedFixture {
    fn name(&self) -> &str {
        "struct_nested.vortex"
    }

    fn description(&self) -> &str {
        "Nested struct: outer struct containing an inner struct with primitive and string fields"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Struct::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let inner = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![
                PrimitiveArray::new(buffer![10i32, 20, 30], Validity::NonNullable).into_array(),
                VarBinArray::from(vec!["x", "y", "z"]).into_array(),
            ],
            3,
            Validity::NonNullable,
        )?;

        let arr = StructArray::try_new(
            FieldNames::from(["inner", "value"]),
            vec![
                inner.into_array(),
                PrimitiveArray::new(buffer![1.1f64, 2.2, 3.3], Validity::NonNullable).into_array(),
            ],
            3,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}

struct ChunkedFixture;

impl ArrayFixture for ChunkedFixture {
    fn name(&self) -> &str {
        "chunked.vortex"
    }

    fn description(&self) -> &str {
        "ChunkedArray with 3 chunks of 1000 rows each containing deterministic u32 values"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Primitive::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let value_gen = |chunk_idx| {
            let values: Vec<u32> = (0u32..1000).map(|i| chunk_idx * 1000 + i).collect();
            let primitives =
                PrimitiveArray::new(vortex_buffer::Buffer::from(values), Validity::NonNullable);
            Ok(StructArray::try_new(
                FieldNames::from(["id"]),
                vec![primitives.into_array()],
                1000,
                Validity::NonNullable,
            )?
            .into_array())
        };

        // 3 chunks of 1000 rows each. Values are deterministic: chunk_idx * 1000 + row_idx.
        Ok(
            ChunkedArray::from_iter((0u32..3).map(value_gen).collect::<VortexResult<Vec<_>>>()?)
                .into_array(),
        )
    }
}

/// All synthetic fixtures. Structs are module-private, so adding a new fixture struct without
/// including it here will produce a dead-code warning from the compiler.
pub fn fixtures() -> Vec<Box<dyn ArrayFixture>> {
    vec![
        Box::new(PrimitivesFixture),
        Box::new(VarBinFixture),
        Box::new(VarBinViewFixture),
        Box::new(BooleansFixture),
        Box::new(StructNestedFixture),
        Box::new(ChunkedFixture),
    ]
}
