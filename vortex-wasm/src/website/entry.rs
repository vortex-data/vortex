// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability::NonNullable;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::scalar::Scalar;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_array::Array;
use vortex_array::ToCanonical;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::website::commit_id::CommitId;
use crate::website::commit_id::PassthroughBuildHasher;

// TODO(connor): Replace with a better `HashMap` to serialize to JavaScript.

/// Maps [`CommitId`] to benchmark value.
pub type CommitValueMap<'a> = HashMap<&'a CommitId, u64, PassthroughBuildHasher>;

/// Maps series name to commit values.
pub type SeriesMap<'a> = HashMap<&'a str, CommitValueMap<'a>>;

/// Maps chart name to series.
pub type ChartMap<'a> = HashMap<&'a str, SeriesMap<'a>>;

/// Maps benchmark group to charts.
pub type GroupedEntries<'a> = HashMap<&'a str, ChartMap<'a>>;

/// A benchmark entry, grouped by benchmark group, then chart name, then series name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    pub commit_id: CommitId,
    pub benchmark_group: String,
    pub chart_name: String,
    pub series_name: String,
    pub value: u64,
}

impl BenchmarkEntry {
    pub fn new(
        commit_id: CommitId,
        benchmark_group: String,
        chart_name: String,
        series_name: String,
        value: u64,
    ) -> Self {
        Self {
            commit_id,
            benchmark_group,
            chart_name,
            series_name,
            value,
        }
    }

    /// Returns the [`DType`] for a [`BenchmarkEntry`].
    ///
    /// The schema is:
    /// - `commit_id`: `FixedSizeList<u8, 20>` (20-byte binary SHA-1)
    /// - `benchmark_group`: `Utf8`
    /// - `chart_name`: `Utf8`
    /// - `series_name`: `Utf8`
    /// - `value`: `u64`
    pub fn dtype() -> DType {
        DType::Struct(
            StructFields::new(
                FieldNames::from([
                    "commit_id",
                    "benchmark_group",
                    "chart_name",
                    "series_name",
                    "value",
                ]),
                vec![
                    DType::FixedSizeList(
                        Arc::new(DType::Primitive(PType::U8, NonNullable)),
                        20,
                        NonNullable,
                    ),
                    DType::Utf8(NonNullable),
                    DType::Utf8(NonNullable),
                    DType::Utf8(NonNullable),
                    DType::Primitive(PType::U64, NonNullable),
                ],
            ),
            NonNullable,
        )
    }

    /// Converts a [`BenchmarkEntry`] to a [`Scalar`].
    pub fn into_scalar(&self) -> Scalar {
        let u8_dtype = DType::Primitive(PType::U8, NonNullable);

        // Convert the 20-byte commit_id to a FixedSizeList scalar.
        let commit_id_bytes: Vec<Scalar> = self
            .commit_id
            .0
            .iter()
            .map(|&b| Scalar::primitive(b, NonNullable))
            .collect();
        let commit_id_scalar = Scalar::fixed_size_list(u8_dtype, commit_id_bytes, NonNullable);

        Scalar::struct_(
            BenchmarkEntry::dtype(),
            vec![
                commit_id_scalar,
                Scalar::utf8(self.benchmark_group.as_str(), NonNullable),
                Scalar::utf8(self.chart_name.as_str(), NonNullable),
                Scalar::utf8(self.series_name.as_str(), NonNullable),
                Scalar::primitive(self.value, NonNullable),
            ],
        )
    }

    /// Converts a Vortex array (expected to be a struct array) into a vector of [`BenchmarkEntry`].
    ///
    /// The array must have the following schema:
    /// - `commit_id`: FixedSizeList<u8, 20>
    /// - `benchmark_group`: Utf8
    /// - `chart_name`: Utf8
    /// - `series_name`: Utf8
    /// - `value`: u64
    pub fn vec_from_array(array: &dyn Array) -> VortexResult<Vec<Self>> {
        // Convert to canonical struct array.
        let struct_array: StructArray = array.to_struct();

        let len = struct_array.len();
        let mut entries = Vec::with_capacity(len);

        // Extract each field.
        let commit_id_field = struct_array.field_by_name("commit_id")?;
        let benchmark_group_field = struct_array.field_by_name("benchmark_group")?;
        let chart_name_field = struct_array.field_by_name("chart_name")?;
        let series_name_field = struct_array.field_by_name("series_name")?;
        let value_field = struct_array.field_by_name("value")?;

        // Convert commit_id to canonical fixed-size list and get the underlying bytes.
        let commit_id_fsl: FixedSizeListArray = commit_id_field.to_fixed_size_list();
        if commit_id_fsl.list_size() != 20 {
            vortex_bail!(
                "Expected commit_id to have list_size 20, got {}",
                commit_id_fsl.list_size()
            );
        }

        // Get the elements as a primitive array of u8.
        let commit_id_elements: PrimitiveArray = commit_id_fsl.elements().to_primitive();
        let commit_id_bytes: &[u8] = commit_id_elements.as_slice();

        // Convert string fields to canonical varbinview arrays.
        let benchmark_group_vbv = benchmark_group_field.to_varbinview();
        let chart_name_vbv = chart_name_field.to_varbinview();
        let series_name_vbv = series_name_field.to_varbinview();

        // Convert value field to primitive array.
        let value_prim: PrimitiveArray = value_field.to_primitive();
        let values: &[u64] = value_prim.as_slice();

        // Build the entries.
        for i in 0..len {
            // Extract the 20-byte commit_id for this row.
            let start = i * 20;
            let end = start + 20;
            let mut commit_id_arr = [0u8; 20];
            commit_id_arr.copy_from_slice(&commit_id_bytes[start..end]);

            // Read strings using bytes_at() and convert to String.
            let benchmark_group = std::str::from_utf8(benchmark_group_vbv.bytes_at(i).as_ref())
                .map_err(|e| vortex_error::vortex_err!("Invalid UTF-8 in benchmark_group: {}", e))?
                .to_string();
            let chart_name = std::str::from_utf8(chart_name_vbv.bytes_at(i).as_ref())
                .map_err(|e| vortex_error::vortex_err!("Invalid UTF-8 in chart_name: {}", e))?
                .to_string();
            let series_name = std::str::from_utf8(series_name_vbv.bytes_at(i).as_ref())
                .map_err(|e| vortex_error::vortex_err!("Invalid UTF-8 in series_name: {}", e))?
                .to_string();

            entries.push(BenchmarkEntry {
                commit_id: CommitId(commit_id_arr),
                benchmark_group,
                chart_name,
                series_name,
                value: values[i],
            });
        }

        Ok(entries)
    }

    /// Groups benchmark entries by benchmark group, chart name, series name, and commit ID.
    pub fn group(entries: &[BenchmarkEntry]) -> GroupedEntries<'_> {
        let mut result: GroupedEntries<'_> = HashMap::new();
        for entry in entries {
            result
                .entry(entry.benchmark_group.as_str())
                .or_default()
                .entry(entry.chart_name.as_str())
                .or_default()
                .entry(entry.series_name.as_str())
                .or_insert_with(|| HashMap::with_hasher(PassthroughBuildHasher))
                .insert(&entry.commit_id, entry.value);
        }
        result
    }
}
