// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::dtype::DType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::partial_ord::partial_max;
use crate::partial_ord::partial_min;
use crate::scalar::Scalar;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Extremum {
    /// Select the least scalar seen so far.
    Min,
    /// Select the greatest scalar seen so far.
    Max,
}

impl Extremum {
    pub(crate) fn stat(self) -> Stat {
        match self {
            Self::Min => Stat::Min,
            Self::Max => Stat::Max,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Min => "vortex.min",
            Self::Max => "vortex.max",
        }
    }

    fn select(self, lhs: Scalar, rhs: Scalar) -> Option<Scalar> {
        match self {
            Self::Min => partial_min(lhs, rhs),
            Self::Max => partial_max(lhs, rhs),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ExtremumPartial {
    /// Current extrema value, stored as a non-null scalar when present.
    value: Option<Scalar>,
    /// Nullable dtype used for the aggregate partial/result scalar.
    element_dtype: DType,
    /// Whether a value-level comparison was unordered.
    ///
    /// This is distinct from dtype support. Unsupported dtypes are rejected by
    /// `return_dtype`/`supports_extrema_dtype`; `unordered` means the dtype was accepted, but
    /// two concrete non-null values could not be ordered by `partial_cmp` (for example a nested
    /// float NaN). Once a partial becomes unordered, the aggregate result must remain unknown
    /// rather than being re-seeded by a later comparable value.
    unordered: bool,
}

impl ExtremumPartial {
    /// Create an empty partial for a min/max aggregate.
    pub(crate) fn new(element_dtype: DType) -> Self {
        Self {
            value: None,
            element_dtype,
            unordered: false,
        }
    }

    /// Merge a scalar into this partial.
    ///
    /// Nulls are ignored. Top-level primitive NaNs are ignored to preserve the existing float
    /// min/max behavior; pruning guards those stats with `NaNCount`. If comparing two accepted
    /// non-null values is unordered, the partial is marked unknown via `unordered`.
    pub(crate) fn merge_scalar(&mut self, extremum: Extremum, scalar: Scalar) -> VortexResult<()> {
        if self.unordered || scalar.is_null() {
            return Ok(());
        }

        if scalar
            .as_primitive_opt()
            .is_some_and(|primitive_scalar| primitive_scalar.is_nan())
        {
            return Ok(());
        }

        let scalar = scalar.cast(&scalar.dtype().as_nonnullable())?;
        self.value = match self.value.take() {
            None => Some(scalar),
            Some(current) => match extremum.select(scalar, current) {
                Some(value) => Some(value),
                None => {
                    self.unordered = true;
                    None
                }
            },
        };

        Ok(())
    }

    /// Convert the partial state to the aggregate scalar.
    ///
    /// Empty input and unordered values both materialize as a typed null scalar, representing an
    /// unknown min/max statistic.
    pub(crate) fn to_scalar(&self) -> VortexResult<Scalar> {
        if self.unordered {
            return Ok(Scalar::null(self.element_dtype.clone()));
        }

        let Some(value) = &self.value else {
            return Ok(Scalar::null(self.element_dtype.clone()));
        };

        value.clone().cast(&self.element_dtype)
    }

    /// Reset the partial so it can be reused for another group.
    pub(crate) fn reset(&mut self) {
        self.value = None;
        self.unordered = false;
    }
}

/// Return the min/max output dtype for an input dtype, or `None` when the dtype is unsupported.
pub(crate) fn extrema_return_dtype(input_dtype: &DType) -> Option<DType> {
    supports_extrema_dtype(input_dtype).then(|| input_dtype.as_nullable())
}

/// Compute an exact min or max statistic for an array.
///
/// This consults and updates the array statistics cache. Unsupported dtypes and unordered values
/// return `None`, which represents an unavailable/unknown statistic.
pub(crate) fn compute_extremum<V>(
    extremum: Extremum,
    vtable: V,
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>>
where
    V: AggregateFnVTable<Options = EmptyOptions>,
{
    if let Precision::Exact(value) = array.statistics().get(extremum.stat()) {
        return Ok(Some(value.cast(&array.dtype().as_nonnullable())?));
    }

    if array.is_empty() {
        return Ok(None);
    }

    if !supports_extrema_dtype(array.dtype()) {
        return Ok(None);
    }

    let mut accumulator = Accumulator::try_new(vtable, EmptyOptions, array.dtype().clone())?;
    accumulator.accumulate(array, ctx)?;
    let value = accumulator.finish()?;
    if value.is_null() {
        return Ok(None);
    }

    let value = value.cast(&array.dtype().as_nonnullable())?;

    if let Some(scalar_value) = value.value() {
        array
            .statistics()
            .set(extremum.stat(), Precision::Exact(scalar_value.clone()));
    }

    Ok(Some(value))
}

/// Accumulate a canonical or constant batch into a min/max partial.
pub(crate) fn accumulate_extremum(
    extremum: Extremum,
    partial: &mut ExtremumPartial,
    batch: &Columnar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if !supports_extrema_dtype(batch.dtype()) {
        vortex_bail!(
            "Aggregate function {} does not support input dtype {}",
            extremum.name(),
            batch.dtype()
        );
    }

    match batch {
        Columnar::Canonical(canonical) => {
            let array = canonical.clone().into_array();
            accumulate_array_extremum(extremum, partial, &array, ctx)
        }
        Columnar::Constant(constant) => {
            if constant.is_empty() {
                Ok(())
            } else {
                partial.merge_scalar(extremum, constant.scalar().clone())
            }
        }
    }
}

fn accumulate_array_extremum(
    extremum: Extremum,
    partial: &mut ExtremumPartial,
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    for idx in 0..array.len() {
        partial.merge_scalar(extremum, array.execute_scalar(idx, ctx)?)?;
    }

    Ok(())
}

/// Return whether min/max is defined for this dtype.
///
/// This is a type-level predicate only. It answers whether the aggregate can be attempted for the
/// dtype. Individual values can still be unordered at runtime; those are tracked on the partial.
pub(crate) fn supports_extrema_dtype(dtype: &DType) -> bool {
    match dtype {
        DType::Null => false,
        DType::Bool(_) => true,
        DType::Primitive(..) => true,
        DType::Decimal(..) => true,
        DType::Utf8(_) => true,
        DType::Binary(_) => true,
        DType::Struct(fields, _) => fields
            .fields()
            .all(|field_dtype| supports_extrema_dtype(&field_dtype)),
        DType::List(element_dtype, _) => supports_extrema_dtype(element_dtype),
        DType::FixedSizeList(element_dtype, ..) => supports_extrema_dtype(element_dtype),
        DType::Union(_) => false,
        DType::Variant(_) => false,
        DType::Extension(extension_dtype) => {
            supports_extrema_dtype(extension_dtype.storage_dtype())
        }
    }
}
