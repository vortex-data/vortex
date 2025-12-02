// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability::NonNullable;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::scalar::Scalar;

/// A benchmark entry, grouped by benchmark group, then chart name, then series name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    pub commit_id: CommitId,
    pub benchmark_group: NameId,
    pub chart_name: NameId,
    pub series_name: NameId,
    pub value: u64,
}

impl BenchmarkEntry {
    pub fn new(
        commit_id: CommitId,
        benchmark_group: NameId,
        chart_name: NameId,
        series_name: NameId,
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
    /// - `benchmark_group`: `u32`
    /// - `chart_name`: `u32`
    /// - `series_name`: `u32`
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
                    DType::Primitive(PType::U32, NonNullable),
                    DType::Primitive(PType::U32, NonNullable),
                    DType::Primitive(PType::U32, NonNullable),
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
                Scalar::primitive(self.benchmark_group.0, NonNullable),
                Scalar::primitive(self.chart_name.0, NonNullable),
                Scalar::primitive(self.series_name.0, NonNullable),
                Scalar::primitive(self.value, NonNullable),
            ],
        )
    }
}

/// String ID lookup so that we don't have to store the string every time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NameId(pub u32);

/// The 20-byte binary SHA-1 Git commit ID.
#[derive(Clone, PartialEq, Eq)]
pub struct CommitId(pub [u8; 20]);

impl fmt::Display for CommitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl fmt::Debug for CommitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CommitId(\"{}\")", hex::encode(self.0))
    }
}

impl Serialize for CommitId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&hex::encode(self.0))
    }
}

impl<'de> Deserialize<'de> for CommitId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct CommitIdVisitor;

        impl<'de> serde::de::Visitor<'de> for CommitIdVisitor {
            type Value = CommitId;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a 40-character hexadecimal string")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value.len() != 40 {
                    return Err(E::custom(format!(
                        "expected 40 hex characters, got {}",
                        value.len()
                    )));
                }

                let bytes = hex::decode(value)
                    .map_err(|e| E::custom(format!("invalid hexadecimal: {}", e)))?;

                let mut arr = [0u8; 20];
                arr.copy_from_slice(&bytes);
                Ok(CommitId(arr))
            }
        }

        deserializer.deserialize_str(CommitIdVisitor)
    }
}
