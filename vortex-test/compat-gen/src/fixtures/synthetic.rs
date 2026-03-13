// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::Fixture;

pub struct PrimitivesFixture;

impl Fixture for PrimitivesFixture {
    fn name(&self) -> &str {
        "primitives.vortex"
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let arr = StructArray::try_new(
            FieldNames::from(["u8", "u16", "u32", "u64", "i32", "i64", "f32", "f64"]),
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
            ],
            3,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

pub struct StringsFixture;

impl Fixture for StringsFixture {
    fn name(&self) -> &str {
        "strings.vortex"
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let strings = VarBinArray::from(vec!["", "hello", "こんにちは", "\u{1f980}"]);
        let arr = StructArray::try_new(
            FieldNames::from(["text"]),
            vec![strings.into_array()],
            4,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

pub struct BooleansFixture;

impl Fixture for BooleansFixture {
    fn name(&self) -> &str {
        "booleans.vortex"
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let bools = BoolArray::from_iter([true, false, true, true, false]);
        let arr = StructArray::try_new(
            FieldNames::from(["flag"]),
            vec![bools.into_array()],
            5,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

pub struct NullableFixture;

impl Fixture for NullableFixture {
    fn name(&self) -> &str {
        "nullable.vortex"
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        let nullable_ints =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(42), None, Some(-7)]);
        let nullable_strings =
            VarBinArray::from(vec![Some("hello"), None, Some("world"), Some(""), None]);
        let arr = StructArray::try_new(
            FieldNames::from(["int_col", "str_col"]),
            vec![nullable_ints.into_array(), nullable_strings.into_array()],
            5,
            Validity::NonNullable,
        )?;
        Ok(vec![arr.into_array()])
    }
}

pub struct StructNestedFixture;

impl Fixture for StructNestedFixture {
    fn name(&self) -> &str {
        "struct_nested.vortex"
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
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
        Ok(vec![arr.into_array()])
    }
}

pub struct ChunkedFixture;

impl Fixture for ChunkedFixture {
    fn name(&self) -> &str {
        "chunked.vortex"
    }

    fn build(&self, _tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>> {
        // 3 chunks of 1000 rows each. Values are deterministic: chunk_idx * 1000 + row_idx.
        (0u32..3)
            .map(|chunk_idx| {
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
            })
            .collect()
    }
}
