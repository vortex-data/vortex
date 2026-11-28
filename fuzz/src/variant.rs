// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Unstructured;
use parquet_variant::Variant;
use parquet_variant::VariantBuilder;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_parquet_variant::ParquetVariant;

/// Encoded unshredded row. An absent row and a present Variant null have different validity.
pub(crate) struct VariantRow {
    valid: bool,
    metadata: Vec<u8>,
    value: Vec<u8>,
}

impl VariantRow {
    pub(crate) fn new(row: Option<Variant<'_, '_>>) -> Self {
        let valid = row.is_some();
        let mut builder = VariantBuilder::new();
        builder.append_value(row.unwrap_or(Variant::Null));
        let (metadata, value) = builder.finish();
        Self {
            valid,
            metadata,
            value,
        }
    }

    pub(crate) fn arbitrary_rows(
        u: &mut Unstructured<'_>,
        nullability: Nullability,
        len: usize,
    ) -> arbitrary::Result<Vec<Self>> {
        (0..len)
            .map(|_| {
                let valid = !nullability.is_nullable() || u.arbitrary::<bool>()?;
                let text: String;
                let row = match u.int_in_range(0u8..=4)? {
                    _ if !valid => None,
                    0 => Some(Variant::Null),
                    1 => Some(Variant::from(u.arbitrary::<bool>()?)),
                    2 => Some(Variant::from(u.arbitrary::<i64>()?)),
                    3 => Some(Variant::from(u.arbitrary::<f64>()?)),
                    _ => {
                        text = u.arbitrary()?;
                        Some(Variant::from(text.as_str()))
                    }
                };
                Ok(Self::new(row))
            })
            .collect()
    }

    pub(crate) fn array<'a>(
        rows: impl Iterator<Item = &'a Self> + Clone,
        nullability: Nullability,
    ) -> ArrayRef {
        let validity = match nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::from_iter(rows.clone().map(|r| r.valid)),
        };
        let metadata = VarBinViewArray::from_iter_bin(rows.clone().map(|r| &r.metadata));
        let value = VarBinViewArray::from_iter_bin(rows.map(|r| &r.value));
        ParquetVariant::try_new(
            validity,
            metadata.into_array(),
            Some(value.into_array()),
            None,
        )
        .vortex_expect("encoded Parquet Variant rows form valid unshredded storage")
        .into_array()
    }
}
