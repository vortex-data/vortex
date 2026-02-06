// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Schema definition for the PolarSignals benchmark.
//!
//! The target Arrow schema (`STACKTRACES_SCHEMA`) is a simplified version of the
//! production PolarSignals writer defined in `pkg/profile/schema.go`. Labels are
//! reduced to 10 representative fields (from 84) covering five fill-rate tiers.
//! RunEndEncoded wrappers are stripped (not yet supported in vortex Arrow
//! execution); Dictionary types are preserved.

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::builder::GenericByteDictionaryBuilder;
use arrow_array::builder::PrimitiveDictionaryBuilder;
use arrow_array::types::Int64Type;
use arrow_array::types::UInt32Type;
use arrow_array::types::UInt64Type;
use arrow_array::types::Utf8Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use arrow_schema::TimeUnit;

/// Label definitions: (field_name, fill_rate, num_distinct) reduced to 10
/// representative labels covering five fill-rate tiers and a range of
/// cardinalities.
///
/// Tiers:
///   - Always filled (100%): cpu, node, thread_name
///   - Near-full (~99%):     comm (queried in Q2)
///   - Mostly filled (~97%): container, namespace
///   - Sparse (~1%):         app, k8s_app
///   - Always null (0%):     action, instance
pub(super) const LABELS: &[(&str, f64, usize)] = &[
    ("action", 0.0, 0),
    ("app", 0.01, 4),
    ("comm", 0.99, 800),
    ("container", 0.97, 50),
    ("cpu", 1.0, 32),
    ("instance", 0.0, 0),
    ("k8s_app", 0.01, 7),
    ("namespace", 0.97, 14),
    ("node", 1.0, 30),
    ("thread_name", 1.0, 390),
];

/// Shorthand type aliases matching production builder types.
pub(super) type StringDictBuilder = GenericByteDictionaryBuilder<UInt32Type, Utf8Type>;
pub(super) type Int64DictBuilder = PrimitiveDictionaryBuilder<UInt32Type, Int64Type>;
pub(super) type UInt64DictBuilder = PrimitiveDictionaryBuilder<UInt32Type, UInt64Type>;

/// Helper: `Dictionary(UInt32, value_type)`.
pub(super) fn dict_u32(value_type: DataType) -> DataType {
    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(value_type))
}

// ── Schema helpers matching production field order from schema.go ────────────

pub(super) fn label_fields() -> Fields {
    LABELS
        .iter()
        .map(|(name, ..)| Field::new(*name, dict_u32(DataType::Utf8), true))
        .collect()
}

/// Production line struct field order: line, function_name, function_system_name,
/// function_filename, function_start_line.
pub(super) fn lines_fields() -> Fields {
    vec![
        Field::new("line", dict_u32(DataType::Int64), true),
        Field::new("function_name", dict_u32(DataType::Utf8), true),
        Field::new("function_system_name", dict_u32(DataType::Utf8), true),
        Field::new("function_filename", dict_u32(DataType::Utf8), true),
        Field::new("function_start_line", dict_u32(DataType::Int64), true),
    ]
    .into()
}

/// Production location struct field order: address, frame_type, mapping_file,
/// mapping_build_id, lines.
pub(super) fn location_fields() -> Fields {
    vec![
        Field::new("address", dict_u32(DataType::UInt64), true),
        Field::new("frame_type", dict_u32(DataType::Utf8), true),
        Field::new("mapping_file", dict_u32(DataType::Utf8), true),
        Field::new("mapping_build_id", dict_u32(DataType::Utf8), true),
        Field::new(
            "lines",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(lines_fields()),
                true,
            ))),
            true,
        ),
    ]
    .into()
}

/// Target Arrow schema matching production `StructuredStacktraceSchema` from
/// `pkg/profile/schema.go`. REE wrappers are stripped (unsupported in vortex
/// Arrow execution); Dictionary types are preserved.
///
/// Field order: labels, locations, value, producer, sample_type, sample_unit,
/// period_type, period_unit, temporality, period, duration, time_nanos.
pub static STACKTRACES_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    Schema::new(vec![
        Field::new("labels", DataType::Struct(label_fields()), false),
        Field::new(
            "locations",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(location_fields()),
                true,
            ))),
            true,
        ),
        Field::new("value", DataType::Int64, false),
        Field::new("producer", dict_u32(DataType::Utf8), false),
        Field::new("sample_type", dict_u32(DataType::Utf8), false),
        Field::new("sample_unit", dict_u32(DataType::Utf8), false),
        Field::new("period_type", dict_u32(DataType::Utf8), false),
        Field::new("period_unit", dict_u32(DataType::Utf8), false),
        Field::new("temporality", dict_u32(DataType::Utf8), true),
        Field::new("period", DataType::Int64, false),
        Field::new("duration", DataType::Int64, false),
        Field::new(
            "time_nanos",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            false,
        ),
    ])
});
