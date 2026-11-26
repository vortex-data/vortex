// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression scheme using byte-part decomposition.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::WidenArray;
use vortex_array::dtype::DecimalType;
pub use vortex_compressor::builtins::DecimalDictScheme;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsSlots;
use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_decimal_byte_parts::lower_part_count;
use vortex_decimal_byte_parts::max_lower_parts;
use vortex_decimal_byte_parts::split_decimal;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum DecimalSchemeMode {
    V1,
    V2,
}

/// The decimal scheme that only produces v1-compatible DBP arrays (does not compress wide
/// decimals).
pub(crate) static DECIMAL_V1: DecimalScheme = DecimalScheme::v1();

/// The decimal scheme that produces both v1 and v2 compatible DBP arrays (compresses wide
/// decimals).
pub(crate) static DECIMAL_V2: DecimalScheme = DecimalScheme::v2();

/// Compression scheme for decimal arrays via byte-part decomposition.
///
/// Uses the [part layout](vortex_decimal_byte_parts) and wraps each MSP whose valid values fit a
/// narrower integer in an integer [`WidenArray`], including MSPs next to lower parts.
///
/// The v1 mode leaves an array canonical when a valid value needs lower parts. Single-part
/// arrays serialize as v1 in either mode, while arrays with lower parts serialize as v2.
///
/// The default uses v1. [`refine`](Scheme::refine) picks v2 when the v2 ID is allowed and v1
/// otherwise.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DecimalScheme {
    mode: DecimalSchemeMode,
}

impl DecimalScheme {
    /// Creates a decimal scheme configured for v1, disallowing splitting of wide decimals.
    ///
    /// Arrays with a valid value wider than `i64` stay canonical.
    pub const fn v1() -> Self {
        Self {
            mode: DecimalSchemeMode::V1,
        }
    }

    /// Creates a decimal scheme configured for v2, allowing splitting of wide decimals.
    pub const fn v2() -> Self {
        Self {
            mode: DecimalSchemeMode::V2,
        }
    }
}

impl Default for DecimalScheme {
    fn default() -> Self {
        Self::v1()
    }
}

impl Scheme for DecimalScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.decimal.byte_parts"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Decimal(_))
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        match self.mode {
            DecimalSchemeMode::V1 => vec![decimal_byte_parts_v1_id()],
            DecimalSchemeMode::V2 => {
                vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]
            }
        }
    }

    fn refine(&self, allowed: &dyn Fn(&ArrayId) -> bool) -> &dyn Scheme {
        if allowed(&decimal_byte_parts_v2_id()) {
            &DECIMAL_V2
        } else {
            &DECIMAL_V1
        }
    }

    /// Children: msp=0, then up to three lower parts in v2 mode.
    fn num_children(&self) -> usize {
        match self.mode {
            DecimalSchemeMode::V1 => 1,
            DecimalSchemeMode::V2 => 4,
        }
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // v1 leaves the array canonical when a valid value needs lower parts. The sample can miss
        // such a value, so check the full array. Then a scheme that can encode the array, such as
        // `DecimalDictScheme`, does not lose to a sample estimate that v1 cannot keep.
        //
        // An error here causes no skip, because `compress` gets the same error and returns it.
        if self.mode == DecimalSchemeMode::V1
            && data.array_as_decimal().values_type() > DecimalType::I64
            && matches!(needs_lower_parts(data, exec_ctx), Ok(true))
        {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Decimal compression is almost always beneficial (MSP narrowing + primitive compression).
        // As the default for decimals, it still competes with `DecimalDictScheme` when the
        // dictionary has an estimate.
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let decimal = data.array().clone().execute::<DecimalArray>(exec_ctx)?;
        let mask = decimal.validity()?.execute_mask(decimal.len(), exec_ctx)?;
        let lower_part_count = lower_part_count(&decimal, &mask)?;

        // v1 cannot carry lower parts. Returning the input leaves it canonical, because the cascade
        // keeps its input when the output is not smaller.
        if self.mode == DecimalSchemeMode::V1 && lower_part_count > 0 {
            // Only a valid value outside the precision needs more lower parts than the precision
            // allows. `split_decimal` rejects that count, but v1 returns before the split.
            let decimal_dtype = decimal.decimal_dtype();
            debug_assert!(
                lower_part_count <= max_lower_parts(decimal_dtype),
                "{decimal_dtype} allows at most {} lower parts, got {lower_part_count}",
                max_lower_parts(decimal_dtype),
            );

            return Ok(data.array().clone());
        }

        let parts = split_decimal(&decimal, &mask, lower_part_count)?;

        let canonical_msp = parts.msp.execute::<PrimitiveArray>(exec_ctx)?;
        let msp = WidenArray::encode(canonical_msp, exec_ctx)?;

        // NB: Compress the narrow child, then wrap it. `CascadingCompressor::compress_child`
        // executes its input to canonical form, which would undo the `Widen`.
        let logical_dtype = msp.dtype().clone();
        let compressed_msp = compressor.compress_child(
            &WidenArray::to_stored_integer(&msp),
            &compress_ctx,
            self.id(),
            DecimalBytePartsSlots::MSP,
            exec_ctx,
        )?;
        let msp = WidenArray::from_stored_integer(compressed_msp, logical_dtype)?;

        let lower_parts = parts
            .lower_parts
            .iter()
            .enumerate()
            .map(|(idx, part)| {
                compressor.compress_child(
                    part,
                    &compress_ctx,
                    self.id(),
                    DecimalBytePartsSlots::LOWER_PARTS_OFFSET + idx,
                    exec_ctx,
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;

        DecimalByteParts::try_new_with_lower_parts(msp, lower_parts, decimal.decimal_dtype())
            .map(IntoArray::into_array)
    }
}

/// Returns `true` if a valid value of the full array is wider than `i64`.
fn needs_lower_parts(data: &ArrayAndStats, exec_ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    let decimal = data.array().clone().execute::<DecimalArray>(exec_ctx)?;
    let mask = decimal.validity()?.execute_mask(decimal.len(), exec_ctx)?;

    Ok(lower_part_count(&decimal, &mask)? > 0)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::Dict;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_compressor::CascadingCompressor;
    use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
    use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
    use vortex_error::VortexResult;

    use super::DECIMAL_V1;
    use super::DECIMAL_V2;
    use super::DecimalDictScheme;
    use crate::SESSION;
    use crate::Scheme;
    use crate::SchemeExt;
    use crate::schemes::integer::BITPACKING_V1;

    /// A value wider than 64 bits that the sample does not hold must not let v1 byte parts win.
    ///
    /// The sample holds only 0 and 1, which byte parts pack at one bit per value. On the full
    /// array, v1 byte parts cannot encode the wide value and returns the array unchanged. The
    /// dictionary holds the wide value, so it must compress the array.
    #[test]
    fn v1_unsampled_wide_value_uses_dict() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        // The deterministic sample does not include the last row.
        let len = 8192;
        let values = (0..len)
            .map(|i| {
                if i == len - 1 {
                    1i128 << 100
                } else {
                    (i % 2) as i128
                }
            })
            .collect::<Buffer<i128>>();
        let array = DecimalArray::new(
            values,
            DecimalDType::new(38, 0),
            Validity::NonNullable,
            &mut ctx,
        )
        .into_array();

        let compressor =
            CascadingCompressor::new(vec![&DecimalDictScheme, &DECIMAL_V1, &BITPACKING_V1]);
        let compressed = compressor.compress(&array, &mut ctx)?;

        assert!(compressed.is::<Dict>(), "{}", compressed.display_tree());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    /// Both variants refine to v2 exactly when the v2 ID is allowed.
    #[rstest]
    #[case::neither(false, false, false)]
    #[case::v1(true, false, false)]
    #[case::v2_only(false, true, true)]
    #[case::both(true, true, true)]
    fn refine_picks_v2_when_v2_id_is_allowed(
        #[case] allow_v1: bool,
        #[case] allow_v2: bool,
        #[case] expect_v2: bool,
        #[values(&DECIMAL_V1, &DECIMAL_V2)] scheme: &'static dyn Scheme,
    ) {
        let allowed = |id: &ArrayId| {
            (allow_v1 && *id == decimal_byte_parts_v1_id())
                || (allow_v2 && *id == decimal_byte_parts_v2_id())
        };
        let refined = scheme.refine(&allowed);
        assert_eq!(refined.id(), scheme.id());
        let expected = if expect_v2 {
            vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]
        } else {
            vec![decimal_byte_parts_v1_id()]
        };
        assert_eq!(refined.produced_encodings(), expected);
    }
}
