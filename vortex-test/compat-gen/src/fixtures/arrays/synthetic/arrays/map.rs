// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::Map;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct MapFixture;

impl FlatLayoutFixture for MapFixture {
    fn name(&self) -> &str {
        "map.vortex"
    }

    fn description(&self) -> &str {
        "Map arrays with unsorted and sorted keys, nullable values, and null maps"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Map.id()]
    }

    fn build(&self, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        // map<i32, utf8?> with keys_sorted = false:
        //   [{1: "one", 2: null}, {}, {1: "dup-old", 1: "dup-new"}, {5: "five"}]
        //
        // Row 2 repeats key 1 on purpose: duplicate keys are representable and a reader must
        // preserve both entries in order rather than deduplicating them.
        let attrs_keys =
            PrimitiveArray::new(buffer![1i32, 2, 1, 1, 5], Validity::NonNullable).into_array();
        let attrs_values = VarBinArray::from_iter(
            [
                Some("one"),
                None,
                Some("dup-old"),
                Some("dup-new"),
                Some("five"),
            ],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let attrs_entries = StructArray::try_new(
            FieldNames::from(["key", "value"]),
            vec![attrs_keys, attrs_values],
            5,
            Validity::NonNullable,
        )?;
        let attrs = MapArray::try_new(
            MapDType::try_new(
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::Nullable),
                false,
            )?,
            ListViewArray::try_new(
                attrs_entries.into_array(),
                PrimitiveArray::new(buffer![0u32, 2, 2, 4], Validity::NonNullable).into_array(),
                PrimitiveArray::new(buffer![2u32, 0, 2, 1], Validity::NonNullable).into_array(),
                Validity::NonNullable,
            )?,
        )?;

        // map<utf8, i64> with keys_sorted = true:
        //   [{"a": 1, "b": 2}, {"z": 26}, {}, {"k": 11, "m": 13, "n": 14}]
        //
        // Keys are sorted within each row, matching the dtype's sortedness assertion.
        let sorted_keys = VarBinArray::from_iter(
            [
                Some("a"),
                Some("b"),
                Some("z"),
                Some("k"),
                Some("m"),
                Some("n"),
            ],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let sorted_values =
            PrimitiveArray::new(buffer![1i64, 2, 26, 11, 13, 14], Validity::NonNullable)
                .into_array();
        let sorted_entries = StructArray::try_new(
            FieldNames::from(["key", "value"]),
            vec![sorted_keys, sorted_values],
            6,
            Validity::NonNullable,
        )?;
        let sorted_attrs = MapArray::try_new(
            MapDType::try_new(
                DType::Utf8(Nullability::NonNullable),
                DType::Primitive(PType::I64, Nullability::NonNullable),
                true,
            )?,
            ListViewArray::try_new(
                sorted_entries.into_array(),
                PrimitiveArray::new(buffer![0u32, 2, 3, 3], Validity::NonNullable).into_array(),
                PrimitiveArray::new(buffer![2u32, 1, 0, 3], Validity::NonNullable).into_array(),
                Validity::NonNullable,
            )?,
        )?;

        // Nullable map<i32, utf8?>: [null, {}, {7: "seven"}, {8: null}]
        //
        // A null map and an empty map are distinct values, and the last row carries a present
        // key with a null value.
        let nullable_keys =
            PrimitiveArray::new(buffer![7i32, 8], Validity::NonNullable).into_array();
        let nullable_values =
            VarBinArray::from_iter([Some("seven"), None], DType::Utf8(Nullability::Nullable))
                .into_array();
        let nullable_entries = StructArray::try_new(
            FieldNames::from(["key", "value"]),
            vec![nullable_keys, nullable_values],
            2,
            Validity::NonNullable,
        )?;
        let nullable_attrs = MapArray::try_new(
            MapDType::try_new(
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::Nullable),
                false,
            )?,
            ListViewArray::try_new(
                nullable_entries.into_array(),
                PrimitiveArray::new(buffer![0u32, 0, 0, 1], Validity::NonNullable).into_array(),
                PrimitiveArray::new(buffer![0u32, 0, 1, 1], Validity::NonNullable).into_array(),
                Validity::from_iter([false, true, true, true]),
            )?,
        )?;

        let arr = StructArray::try_new(
            FieldNames::from(["attrs", "sorted_attrs", "nullable_attrs"]),
            vec![
                attrs.into_array(),
                sorted_attrs.into_array(),
                nullable_attrs.into_array(),
            ],
            4,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
