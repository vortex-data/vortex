// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constant encoding schemes for integer, float, and string arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexResult;

use super::is_float_primitive;
use super::is_integer_primitive;
use super::is_utf8_string;
use crate::CascadingCompressor;
use crate::ctx::CompressorContext;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;

/// Constant encoding for integer arrays with a single distinct value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IntConstantScheme;

impl Scheme for IntConstantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.constant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
    }

    fn detects_constant(&self) -> bool {
        true
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<f64> {
        if ctx.is_sample() {
            return Ok(0.0);
        }

        let stats = data.integer_stats();

        if stats.distinct_count().is_none_or(|count| count > 1) {
            return Ok(0.0);
        }

        Ok(stats.value_count() as f64)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let source = data.integer_stats().source().clone();
        compress_constant_primitive(&source)
    }
}

/// Constant encoding for float arrays with a single distinct value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatConstantScheme;

impl Scheme for FloatConstantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.constant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_float_primitive(canonical)
    }

    fn detects_constant(&self) -> bool {
        true
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<f64> {
        if ctx.is_sample() {
            return Ok(0.0);
        }

        let stats = data.float_stats();

        if stats.null_count() as usize == stats.source().len() || stats.value_count() == 0 {
            return Ok(0.0);
        }

        if stats.distinct_count().is_some_and(|count| count == 1) {
            return Ok(stats.value_count() as f64);
        }

        Ok(0.0)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let source = data.float_stats().source().clone();
        compress_constant_primitive(&source)
    }
}

/// Constant encoding for string arrays with a single distinct value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StringConstantScheme;

impl Scheme for StringConstantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.constant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_utf8_string(canonical)
    }

    fn detects_constant(&self) -> bool {
        true
    }

    fn expected_compression_ratio(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<f64> {
        if ctx.is_sample() {
            return Ok(0.0);
        }

        let stats = data.string_stats();

        if stats.estimated_distinct_count().is_none_or(|c| c > 1)
            || !is_constant(
                &stats.source().clone().into_array(),
                &mut compressor.execution_ctx(),
            )?
        {
            return Ok(0.0);
        }

        // Force constant in these cases.
        Ok(f64::MAX)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let stats = data.string_stats();

        let scalar_idx =
            (0..stats.source().len()).position(|idx| stats.source().is_valid(idx).unwrap_or(false));

        match scalar_idx {
            Some(idx) => {
                let scalar = stats.source().scalar_at(idx)?;
                let const_arr = ConstantArray::new(scalar, stats.source().len()).into_array();
                if !stats.source().all_valid()? {
                    Ok(
                        MaskedArray::try_new(const_arr, stats.source().validity().clone())?
                            .into_array(),
                    )
                } else {
                    Ok(const_arr)
                }
            }
            None => Ok(ConstantArray::new(
                Scalar::null(stats.source().dtype().clone()),
                stats.source().len(),
            )
            .into_array()),
        }
    }
}

/// Shared helper for compressing a constant primitive array (int or float).
fn compress_constant_primitive(source: &PrimitiveArray) -> VortexResult<ArrayRef> {
    let scalar_idx = (0..source.len()).position(|idx| source.is_valid(idx).unwrap_or(false));

    match scalar_idx {
        Some(idx) => {
            let scalar = source.scalar_at(idx)?;
            let const_arr = ConstantArray::new(scalar, source.len()).into_array();
            if !source.all_valid()? {
                Ok(MaskedArray::try_new(const_arr, source.validity().clone())?.into_array())
            } else {
                Ok(const_arr)
            }
        }
        None => {
            Ok(ConstantArray::new(Scalar::null(source.dtype().clone()), source.len()).into_array())
        }
    }
}
