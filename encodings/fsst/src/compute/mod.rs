// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod byte_length;
mod cast;
mod compare;
mod filter;
mod like;
pub(crate) mod uncompressed_size_in_bytes;

use num_traits::ToPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbin::take_varbin;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::IntegerPType;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProviderExt;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use crate::FSST;
use crate::FSSTArrayExt;
use crate::FSSTArraySlotsExt;

/// Small arrays gather only when their compressed payload is large enough to amortize setup.
const MIN_GATHER_LEN: usize = 1024;
/// Minimum compressed payload for gathering from a small array, measured by `fsst_take`.
const MIN_GATHER_COMPRESSED_BYTES: u64 = 8 * 1024;

/// Only takes with fewer indices than `1 / SPARSE_TAKE_DENOMINATOR` of the rows gather compressed
/// rows. A denser take decodes every row once and gathers the decoded strings instead: gathering
/// decodes a row again for every index that repeats it, and costs more per row than decoding
/// in bulk.
const SPARSE_TAKE_DENOMINATOR: usize = 4;

impl TakeExecute for FSST {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if indices.len().saturating_mul(SPARSE_TAKE_DENOMINATOR) >= array.len() {
            return Ok(None);
        }
        if array.len() < MIN_GATHER_LEN {
            let Some(bytes) = compressed_byte_span(array.codes_offsets())
                .filter(|&bytes| bytes >= MIN_GATHER_COMPRESSED_BYTES)
            else {
                return Ok(None);
            };
            // Leave at least half the codes untouched to amortize gathering and the cost check.
            if !take_fits_byte_budget(array.codes_offsets(), indices, bytes / 2, ctx)? {
                return Ok(None);
            }
        }

        Ok(Some(
            FSST::try_new_with_symbol_table(
                array
                    .dtype()
                    .clone()
                    .union_nullability(indices.dtype().nullability()),
                array.symbol_table(),
                take_varbin(array.codes().as_view(), indices, ctx)?,
                array
                    .uncompressed_lengths()
                    .take(indices.clone())?
                    .fill_null(Scalar::zero_value(
                        &array.uncompressed_lengths_dtype().clone(),
                    ))?,
                ctx,
            )?
            .into_array(),
        ))
    }
}

fn take_fits_byte_budget(
    offsets: &ArrayRef,
    indices: &ArrayRef,
    budget: u64,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let offsets = offsets.clone().execute::<PrimitiveArray>(ctx)?;
    let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
    let validity = indices.validity()?.execute_mask(indices.len(), ctx)?;
    // Dispatch only on widths: offsets and valid indices are nonnegative.
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let indices = indices.reinterpret_cast(indices.ptype().to_unsigned());
    Ok(match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        let offsets = offsets.as_slice::<O>();
        match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            let indices = indices.as_slice::<I>();
            match validity.indices() {
                AllOr::All => selected_bytes_fit(offsets, indices.iter().copied(), budget),
                AllOr::None => true,
                AllOr::Some(valid) => {
                    selected_bytes_fit(offsets, valid.iter().map(|&i| indices[i]), budget)
                }
            }
        })
    }))
}

// Count repeats before copying codes: gathering decodes each selected occurrence separately.
fn selected_bytes_fit<O: IntegerPType, I: IntegerPType>(
    offsets: &[O],
    mut indices: impl Iterator<Item = I>,
    mut budget: u64,
) -> bool {
    indices.all(|index| {
        let Some(index) = index.to_usize().filter(|&i| i < offsets.len() - 1) else {
            return false;
        };
        let remaining = offsets[index + 1]
            .to_u64()
            .zip(offsets[index].to_u64())
            .and_then(|(end, start)| end.checked_sub(start))
            .and_then(|len| budget.checked_sub(len));
        let Some(remaining) = remaining else {
            return false;
        };
        budget = remaining;
        true
    })
}

// Slices retain their backing bytes, so only the logical offsets describe their decode cost.
// Inspect already-materialized offsets or cached exact bounds without executing either child.
fn compressed_byte_span(offsets: &ArrayRef) -> Option<u64> {
    if let Some(offsets) = offsets.as_opt::<Primitive>()
        && offsets.buffer_handle().as_host_opt().is_some()
    {
        return match_each_integer_ptype!(offsets.ptype(), |P| {
            let values = offsets.as_slice::<P>();
            let first = values.first()?.to_u64()?;
            let last = values.last()?.to_u64()?;
            last.checked_sub(first)
        });
    }
    let stats = offsets.statistics();
    let first = stats.get_as::<u64>(Stat::Min).as_exact()?;
    let last = stats.get_as::<u64>(Stat::Max).as_exact()?;
    last.checked_sub(first)
}

#[cfg(test)]
mod tests {
    use fsst::Compressor;
    use fsst::Symbol;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::dict::TakeExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::stats::Precision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::compressed_byte_span;
    use crate::FSST;
    use crate::FSSTArray;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    #[test]
    fn test_take_null() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let arr =
            VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable)).into_array();
        let compr = fsst_train_compressor(&arr, &mut ctx)?;
        let fsst = fsst_compress(&arr, &compr, &mut ctx)?;

        let idx1: PrimitiveArray = (0..1).collect();

        assert_eq!(
            fsst.take(idx1.into_array())?.dtype(),
            &DType::Utf8(Nullability::NonNullable)
        );

        let idx2: PrimitiveArray = PrimitiveArray::from_option_iter(vec![Some(0)]);

        assert_eq!(
            fsst.take(idx2.into_array())?.dtype(),
            &DType::Utf8(Nullability::Nullable)
        );
        Ok(())
    }

    #[rstest]
    #[case(VarBinArray::from_iter(
        ["hello world", "testing fsst", "compression test", "data array", "vortex encoding"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(
        [Some("hello"), None, Some("world"), Some("test"), None],
        DType::Utf8(Nullability::Nullable),
    ))]
    #[case(VarBinArray::from_iter(
        ["single element"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    fn test_take_fsst_conformance(#[case] varbin: VarBinArray) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let varbin = varbin.into_array();
        let compressor = fsst_train_compressor(&varbin, &mut ctx)?;
        let array = fsst_compress(&varbin, &compressor, &mut ctx)?;
        test_take_conformance(&array.into_array(), &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(2048, 1, false, true)]
    #[case(2048, 3, true, true)]
    #[case(2048, 4, false, false)]
    #[case(2, 1, false, false)]
    fn test_take_small_byte_gate(
        #[case] repeats: usize,
        #[case] selected: usize,
        #[case] nullable: bool,
        #[case] gathers: bool,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let input = VarBinArray::from_iter(
            (0..16).map(|row| {
                (!(nullable && row == 0)).then(|| format!("{}{row}", "caption ".repeat(repeats)))
            }),
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let compressor = Compressor::rebuild_from([Symbol::from_slice(b"caption ")], [8u8]);
        let array = fsst_compress(&input, &compressor, &mut ctx)?;
        let indices = if nullable {
            PrimitiveArray::from_option_iter([Some(15u64), None, Some(0)]).into_array()
        } else {
            PrimitiveArray::from_iter((0..selected).map(|index| (15 - index) as u64)).into_array()
        };
        assert_eq!(
            FSST::take(array.as_view(), &indices, &mut ctx)?.is_some(),
            gathers
        );
        assert_arrays_eq!(
            array.into_array().take(indices.clone())?,
            input.take(indices)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_small_slice_uses_logical_byte_span() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let input = VarBinArray::from_iter(
            (0..16).map(|row| Some("caption ".repeat(if row < 4 { 4096 } else { 2 }))),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let compressor = Compressor::rebuild_from([Symbol::from_slice(b"caption ")], [8u8]);
        let array = fsst_compress(&input, &compressor, &mut ctx)?.into_array();
        let indices = buffer![0u64].into_array();
        for (range, gathers) in [(0..8, true), (4..16, false)] {
            let sliced = array.slice(range.clone())?;
            assert_eq!(
                FSST::take(sliced.as_::<FSST>(), &indices, &mut ctx)?.is_some(),
                gathers
            );
            assert_arrays_eq!(
                sliced.take(indices.clone())?,
                input.slice(range)?.take(indices.clone())?,
                &mut ctx
            );
        }
        Ok(())
    }

    #[rstest]
    #[case::repeated_long_row(1, PrimitiveArray::from_iter([0u64; 63]).into_array(), false)]
    #[case::distinct_long_rows(16, PrimitiveArray::from_iter(0i32..16).into_array(), false)]
    #[case::short_row_with_null_index(1, PrimitiveArray::new(
        buffer![u64::MAX, 1], Validity::from_iter([false, true]),
    ).into_array(), true)]
    #[case::encoded_indices(1, DictArray::new(
        buffer![0u8, 0].into_array(), buffer![1u32].into_array(),
    ).into_array(), true)]
    fn test_take_small_uneven_lengths(
        #[case] long_rows: usize,
        #[case] indices: ArrayRef,
        #[case] gathers: bool,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let input = VarBinArray::from_iter(
            (0..256).map(|row| Some("caption ".repeat(if row < long_rows { 8192 } else { 1 }))),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let compressor = Compressor::rebuild_from([Symbol::from_slice(b"caption ")], [8u8]);
        let array = fsst_compress(&input, &compressor, &mut ctx)?;
        assert_eq!(
            FSST::take(array.as_view(), &indices, &mut ctx)?.is_some(),
            gathers
        );
        assert_arrays_eq!(
            array.into_array().take(indices.clone())?,
            input.take(indices)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_take_compressed_span_requires_exact_cached_bounds() {
        let offsets = DictArray::new(
            buffer![0u32, 1].into_array(),
            buffer![1000u64, 20000].into_array(),
        )
        .into_array();
        assert_eq!(compressed_byte_span(&offsets), None);
        offsets
            .statistics()
            .set(Stat::Min, Precision::exact(1000u64));
        offsets
            .statistics()
            .set(Stat::Max, Precision::exact(20000u64));
        assert_eq!(compressed_byte_span(&offsets), Some(19000));
        offsets
            .statistics()
            .set(Stat::Min, Precision::Inexact(1000u64.into()));
        assert_eq!(compressed_byte_span(&offsets), None);
    }

    type FsstBuilder = fn(&mut ExecutionCtx) -> FSSTArray;

    #[rstest]
    // Basic string arrays
    #[case::fsst_simple(|ctx: &mut ExecutionCtx| {
        let array = VarBinArray::from_iter(
            ["hello world", "testing fsst", "compression test", "data array", "vortex encoding"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        ).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]
    // Nullable strings
    #[case::fsst_nullable(|ctx: &mut ExecutionCtx| {
        let array = VarBinArray::from_iter(
            [Some("hello"), None, Some("world"), Some("test"), None],
            DType::Utf8(Nullability::Nullable),
        ).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]
    // Repetitive patterns (good for FSST compression)
    #[case::fsst_repetitive(|ctx: &mut ExecutionCtx| {
        let array = VarBinArray::from_iter(
            ["http://example.com", "http://test.com", "http://vortex.dev", "http://data.org"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        ).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]
    // Edge cases
    #[case::fsst_single(|ctx: &mut ExecutionCtx| {
        let array = VarBinArray::from_iter(
            ["single element"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        ).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]
    #[case::fsst_empty_strings(|ctx: &mut ExecutionCtx| {
        let array = VarBinArray::from_iter(
            ["", "test", "", "hello", ""].map(Some),
            DType::Utf8(Nullability::NonNullable),
        ).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]
    // Large arrays
    #[case::fsst_large(|ctx: &mut ExecutionCtx| {
        let data: Vec<Option<&str>> = (0..1500)
            .map(|i| Some(match i % 10 {
                0 => "https://www.example.com/page",
                1 => "https://www.test.org/data",
                2 => "https://www.vortex.dev/docs",
                3 => "https://www.github.com/apache/arrow",
                4 => "https://www.rust-lang.org/learn",
                5 => "SELECT * FROM table WHERE id = ",
                6 => "INSERT INTO users (name, email) VALUES",
                7 => "UPDATE records SET status = 'active'",
                8 => "DELETE FROM logs WHERE timestamp < ",
                _ => "CREATE TABLE data (id INT, value TEXT)",
            }))
            .collect();
        let array = VarBinArray::from_iter(data, DType::Utf8(Nullability::NonNullable)).into_array();
        let compressor = fsst_train_compressor(&array, ctx).unwrap();
        fsst_compress(&array, &compressor, ctx).unwrap()
    })]

    fn test_fsst_consistency(#[case] build: FsstBuilder) {
        let mut ctx = array_session().create_execution_ctx();
        let array = build(&mut ctx);
        test_array_consistency(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
