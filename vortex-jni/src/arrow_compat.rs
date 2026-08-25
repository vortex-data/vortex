// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixups applied to Arrow data on its way out to Java.
//!
//! arrow-java (19.0.0, pinned in `java/gradle/libs.versions.toml`) implements only part of the
//! C Data Interface, so a few Arrow constructs that Vortex produces cannot cross the boundary
//! as-is. Each fixup here compensates for one such gap and can be deleted when the Java
//! dependency catches up.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::array::make_array;
use arrow_data::ArrayData;
use arrow_data::transform::MutableArrayData;
use arrow_schema::DataType;
use arrow_schema::FieldRef;
use arrow_schema::Fields;
use arrow_schema::Schema;
use vortex::error::VortexResult;

/// Rewrite every `Decimal32` and `Decimal64` in `schema` to `Decimal128`, recursing through
/// nested types.
///
/// arrow-java has no `Decimal32Vector` or `Decimal64Vector`. Its
/// `Types.getMinorTypeForArrowType` maps every `ArrowType.Decimal` whose bit width is not 256
/// onto the 128-bit `DecimalVector`, and the C Data importer then sizes the values buffer at 16
/// bytes per slot. A `Decimal32` or `Decimal64` array is therefore read out of a buffer that is
/// four or two times too small: no error, just wrong values and out-of-bounds reads.
///
/// Widening the schema is enough to fix both halves of the export path, because the array
/// executor is driven by the target Arrow type: given a `Decimal128` target it sign-extends the
/// narrow values into 128-bit lanes.
pub(crate) fn widen_small_decimals(schema: Schema) -> Schema {
    let Some(fields) = widen_fields(&schema.fields) else {
        return schema;
    };

    Schema::new_with_metadata(fields, schema.metadata)
}

/// The widened form of `fields`, or `None` when none of them holds a narrow decimal.
fn widen_fields(fields: &Fields) -> Option<Fields> {
    let mut widened = false;
    let fields = fields
        .iter()
        .map(|field| match widen_field(field) {
            Some(field) => {
                widened = true;
                field
            }
            None => FieldRef::clone(field),
        })
        .collect::<Fields>();

    widened.then_some(fields)
}

/// The widened form of `field`, or `None` when it holds no narrow decimal.
fn widen_field(field: &FieldRef) -> Option<FieldRef> {
    widen_data_type(field.data_type())
        .map(|data_type| FieldRef::new(field.as_ref().clone().with_data_type(data_type)))
}

/// The widened form of `data_type`, or `None` when it holds no narrow decimal.
fn widen_data_type(data_type: &DataType) -> Option<DataType> {
    match data_type {
        DataType::Decimal32(precision, scale) | DataType::Decimal64(precision, scale) => {
            Some(DataType::Decimal128(*precision, *scale))
        }
        DataType::Struct(fields) => widen_fields(fields).map(DataType::Struct),
        DataType::List(field) => widen_field(field).map(DataType::List),
        DataType::LargeList(field) => widen_field(field).map(DataType::LargeList),
        DataType::ListView(field) => widen_field(field).map(DataType::ListView),
        DataType::LargeListView(field) => widen_field(field).map(DataType::LargeListView),
        DataType::FixedSizeList(field, size) => {
            widen_field(field).map(|field| DataType::FixedSizeList(field, *size))
        }
        DataType::Map(entries, keys_sorted) => {
            widen_field(entries).map(|entries| DataType::Map(entries, *keys_sorted))
        }
        DataType::Dictionary(keys, values) => widen_data_type(values)
            .map(|values| DataType::Dictionary(keys.clone(), Box::new(values))),
        DataType::RunEndEncoded(ends, values) => {
            widen_field(values).map(|values| DataType::RunEndEncoded(Arc::clone(ends), values))
        }
        _ => None,
    }
}

/// Copy any column carrying a non-zero array offset into an offset-0 equivalent because
/// arrow-java's C Data importer ignores `offset`.
pub(crate) fn rebase_offsets(batch: RecordBatch) -> VortexResult<RecordBatch> {
    let mut rebased = false;
    let columns = batch
        .columns()
        .iter()
        .map(|column| {
            let data = column.to_data();
            if carries_offset(&data) {
                rebased = true;
                // `concat` of a single array will not do this: it short-circuits to
                // `slice(0, len)`, which keeps the offset.
                let len = data.len();
                let mut copy = MutableArrayData::new(vec![&data], false, len);
                copy.try_extend(0, 0, len)?;
                Ok(make_array(copy.freeze()))
            } else {
                Ok(Arc::clone(column))
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;

    if !rebased {
        return Ok(batch);
    }
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

/// Whether `data` or any of its descendants carries a non-zero offset.
fn carries_offset(data: &ArrayData) -> bool {
    data.offset() != 0 || data.child_data().iter().any(carries_offset)
}

#[cfg(test)]
mod tests {
    use arrow_schema::Field;
    use rstest::rstest;

    use super::*;

    fn widen(data_type: DataType) -> DataType {
        let schema = Schema::new(vec![Field::new("col", data_type, true)]);
        widen_small_decimals(schema).field(0).data_type().clone()
    }

    #[rstest]
    #[case(DataType::Decimal32(9, 2), DataType::Decimal128(9, 2))]
    #[case(DataType::Decimal64(18, 4), DataType::Decimal128(18, 4))]
    #[case(DataType::Decimal128(38, 2), DataType::Decimal128(38, 2))]
    #[case(DataType::Decimal256(39, 2), DataType::Decimal256(39, 2))]
    #[case(DataType::Int64, DataType::Int64)]
    fn widens_narrow_decimals(#[case] data_type: DataType, #[case] expected: DataType) {
        assert_eq!(widen(data_type), expected);
    }

    #[test]
    fn widens_decimals_nested_in_containers() {
        let decimal = Field::new("item", DataType::Decimal64(15, 2), false);
        let widened = Field::new("item", DataType::Decimal128(15, 2), false);

        assert_eq!(
            widen(DataType::List(Arc::new(decimal.clone()))),
            DataType::List(Arc::new(widened.clone()))
        );
        assert_eq!(
            widen(DataType::Struct(Fields::from(vec![decimal.clone()]))),
            DataType::Struct(Fields::from(vec![widened.clone()]))
        );

        let entries = |value: Field| {
            Arc::new(Field::new_struct(
                "entries",
                Fields::from(vec![Field::new("key", DataType::Utf8, false), value]),
                false,
            ))
        };
        assert_eq!(
            widen(DataType::Map(entries(decimal), false)),
            DataType::Map(entries(widened), false)
        );
    }

    #[test]
    fn keeps_metadata_and_nullability() {
        let field = Field::new("col", DataType::Decimal32(4, 1), true)
            .with_metadata([("k".to_string(), "v".to_string())].into());
        let schema = Schema::new(vec![field])
            .with_metadata([("schema".to_string(), "meta".to_string())].into());

        let widened = widen_small_decimals(schema);

        assert_eq!(widened.metadata().get("schema"), Some(&"meta".to_string()));
        assert_eq!(widened.field(0).metadata().get("k"), Some(&"v".to_string()));
        assert!(widened.field(0).is_nullable());
    }
}
