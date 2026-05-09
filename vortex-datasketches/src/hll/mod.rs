// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod primitive;
mod varbin;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use datasketches::hll::HllSketch as DatasketchesHllSketch;
use datasketches::hll::HllType;
use datasketches::hll::HllUnion;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use self::bool::accumulate_bool;
use self::bool::update_bool;
use self::primitive::accumulate_primitive;
use self::primitive::update_primitive;
use self::varbin::accumulate_varbinview;
use self::varbin::update_binary;
use self::varbin::update_utf8;

/// Default HLL precision (`lg_k`) used by [`HllOptions`].
pub const DEFAULT_LG_K: u8 = 12;

const OPTIONS_VERSION: u8 = 1;

/// Apache DataSketches HLL target representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HllTarget {
    /// HLL_4, the most compact representation.
    Hll4,
    /// HLL_6, a balanced representation.
    Hll6,
    /// HLL_8, the highest precision-byte representation.
    #[default]
    Hll8,
}

impl HllTarget {
    fn as_byte(self) -> u8 {
        match self {
            Self::Hll4 => 4,
            Self::Hll6 => 6,
            Self::Hll8 => 8,
        }
    }

    fn from_byte(byte: u8) -> VortexResult<Self> {
        Ok(match byte {
            4 => Self::Hll4,
            6 => Self::Hll6,
            8 => Self::Hll8,
            _ => vortex_bail!("Invalid HLL target type byte: {byte}"),
        })
    }
}

impl Display for HllTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hll4 => write!(f, "hll4"),
            Self::Hll6 => write!(f, "hll6"),
            Self::Hll8 => write!(f, "hll8"),
        }
    }
}

impl From<HllTarget> for HllType {
    fn from(value: HllTarget) -> Self {
        match value {
            HllTarget::Hll4 => Self::Hll4,
            HllTarget::Hll6 => Self::Hll6,
            HllTarget::Hll8 => Self::Hll8,
        }
    }
}

/// Options for the `datasketches.hll` aggregate function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HllOptions {
    /// Log2 of the configured number of HLL buckets. Apache DataSketches supports `[4, 21]`.
    pub lg_k: u8,
    /// HLL target representation used for serialized aggregate results.
    pub target: HllTarget,
}

impl Default for HllOptions {
    fn default() -> Self {
        Self {
            lg_k: DEFAULT_LG_K,
            target: HllTarget::default(),
        }
    }
}

impl Display for HllOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "lg_k={},target={}", self.lg_k, self.target)
    }
}

impl HllOptions {
    fn validate(&self) -> VortexResult<()> {
        vortex_ensure!(
            (4..=21).contains(&self.lg_k),
            "HLL lg_k must be in [4, 21], got {}",
            self.lg_k
        );
        Ok(())
    }
}

/// Aggregate function vtable for Apache DataSketches HLL sketches.
#[derive(Clone, Debug)]
pub struct Hll;

/// Partial accumulator state for [`Hll`].
pub struct HllPartial {
    options: HllOptions,
    union: HllUnion,
}

impl HllPartial {
    fn update_value<T: std::hash::Hash>(&mut self, value: T) {
        self.union.update_value(value);
    }
}

/// Build a serialized Apache DataSketches HLL sketch from an array.
pub fn hll(array: &ArrayRef, options: HllOptions, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(Hll, options, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    acc.finish()
}

/// Return the cardinality estimate for a serialized HLL sketch.
pub fn estimate(bytes: &[u8]) -> VortexResult<f64> {
    let sketch = DatasketchesHllSketch::deserialize(bytes)
        .map_err(|err| vortex_err!("Failed to deserialize HLL sketch: {err}"))?;
    Ok(sketch.estimate())
}

impl AggregateFnVTable for Hll {
    type Options = HllOptions;
    type Partial = HllPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("datasketches.hll")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        options.validate()?;
        Ok(Some(vec![
            OPTIONS_VERSION,
            options.lg_k,
            options.target.as_byte(),
        ]))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_ensure!(
            metadata.len() == 3,
            "Invalid HLL options metadata length: expected 3, got {}",
            metadata.len()
        );
        vortex_ensure!(
            metadata[0] == OPTIONS_VERSION,
            "Unsupported HLL options metadata version: {}",
            metadata[0]
        );

        let options = HllOptions {
            lg_k: metadata[1],
            target: HllTarget::from_byte(metadata[2])?,
        };
        options.validate()?;
        Ok(options)
    }

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        if options.validate().is_err() {
            return None;
        }
        match input_dtype {
            DType::Null
            | DType::Bool(_)
            | DType::Primitive(..)
            | DType::Utf8(_)
            | DType::Binary(_) => Some(DType::Binary(Nullability::NonNullable)),
            _ => None,
        }
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        options.validate()?;
        Ok(HllPartial {
            options: options.clone(),
            union: HllUnion::new(options.lg_k),
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }

        let bytes = other
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("HLL partial scalar must be non-null"))?;
        let sketch = DatasketchesHllSketch::deserialize(bytes.as_slice())
            .map_err(|err| vortex_err!("Failed to deserialize HLL partial: {err}"))?;
        partial.union.update(&sketch);
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let sketch = partial.union.get_result(partial.options.target.into());
        Ok(Scalar::binary(sketch.serialize(), Nullability::NonNullable))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.union.reset();
    }

    fn is_saturated(&self, _state: &Self::Partial) -> bool {
        false
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match batch {
            Columnar::Constant(c) => {
                if c.is_empty() || c.scalar().is_null() {
                    return Ok(());
                }
                match c.scalar().dtype() {
                    DType::Bool(_) => {
                        update_bool(
                            partial,
                            c.scalar()
                                .as_bool()
                                .value()
                                .ok_or_else(|| vortex_err!("checked non-null bool scalar"))?,
                        );
                    }
                    DType::Primitive(..) => update_primitive(partial, c.scalar().as_primitive())?,
                    DType::Utf8(_) => {
                        let value = c
                            .scalar()
                            .as_utf8()
                            .value()
                            .ok_or_else(|| vortex_err!("checked non-null UTF-8 scalar"))?;
                        update_utf8(partial, value.as_str());
                    }
                    DType::Binary(_) => {
                        let value = c
                            .scalar()
                            .as_binary()
                            .value()
                            .ok_or_else(|| vortex_err!("checked non-null binary scalar"))?;
                        update_binary(partial, value.as_slice());
                    }
                    DType::Null => {}
                    _ => vortex_bail!("Unsupported constant dtype for HLL: {}", c.scalar().dtype()),
                }
                Ok(())
            }
            Columnar::Canonical(c) => match c {
                Canonical::Null(_) => Ok(()),
                Canonical::Bool(array) => accumulate_bool(partial, array, ctx),
                Canonical::Primitive(array) => accumulate_primitive(partial, array, ctx),
                Canonical::VarBinView(array) => accumulate_varbinview(partial, array, ctx),
                Canonical::Decimal(_)
                | Canonical::Extension(_)
                | Canonical::Struct(_)
                | Canonical::List(_)
                | Canonical::FixedSizeList(_)
                | Canonical::Variant(_) => {
                    vortex_bail!("Unsupported canonical type for HLL: {}", c.dtype())
                }
            },
        }
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

#[cfg(test)]
mod tests {
    use datasketches::hll::HllSketch as DatasketchesHllSketch;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;

    fn estimate_scalar(scalar: Scalar) -> VortexResult<f64> {
        let bytes = scalar
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("HLL result must be non-null"))?;
        estimate(bytes.as_slice())
    }

    fn assert_estimate_near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 0.000001,
            "expected estimate near {expected}, got {actual}"
        );
    }

    #[test]
    fn hll_skips_nulls() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(1), Some(2), None])
            .into_array();

        let result = hll(&array, HllOptions::default(), &mut ctx)?;

        assert_estimate_near(estimate_scalar(result)?, 2.0);
        Ok(())
    }

    #[test]
    fn hll_supports_bool_and_varbin() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let bools = BoolArray::from_iter([Some(true), None, Some(false), Some(true)]).into_array();
        assert_estimate_near(
            estimate_scalar(hll(&bools, HllOptions::default(), &mut ctx)?)?,
            2.0,
        );

        let strings = VarBinViewArray::from_iter_nullable_str([
            Some("alpha"),
            None,
            Some("beta"),
            Some("alpha"),
        ])
        .into_array();
        assert_estimate_near(
            estimate_scalar(hll(&strings, HllOptions::default(), &mut ctx)?)?,
            2.0,
        );
        Ok(())
    }

    #[test]
    fn hll_combines_partial_sketches() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let mut partial = Hll.empty_partial(&HllOptions::default(), &dtype)?;

        let mut left = DatasketchesHllSketch::new(DEFAULT_LG_K, HllType::Hll8);
        left.update(1i32);
        left.update(2i32);
        Hll.combine_partials(
            &mut partial,
            Scalar::binary(left.serialize(), Nullability::NonNullable),
        )?;

        let mut right = DatasketchesHllSketch::new(DEFAULT_LG_K, HllType::Hll8);
        right.update(2i32);
        right.update(3i32);
        Hll.combine_partials(
            &mut partial,
            Scalar::binary(right.serialize(), Nullability::NonNullable),
        )?;

        assert_estimate_near(estimate_scalar(Hll.finalize_scalar(&partial)?)?, 3.0);
        Ok(())
    }

    #[test]
    fn hll_hashes_constant_utf8_like_canonical_utf8() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();
        let dtype = DType::Utf8(Nullability::NonNullable);
        let mut partial = Hll.empty_partial(&HllOptions::default(), &dtype)?;

        let constant =
            ConstantArray::new(Scalar::utf8("alpha", Nullability::NonNullable), 5).into_array();
        let constant_sketch = hll(&constant, HllOptions::default(), &mut ctx)?;
        Hll.combine_partials(&mut partial, constant_sketch)?;

        let canonical = VarBinViewArray::from_iter_str(["alpha"]).into_array();
        let canonical_sketch = hll(&canonical, HllOptions::default(), &mut ctx)?;
        Hll.combine_partials(&mut partial, canonical_sketch)?;

        assert_estimate_near(estimate_scalar(Hll.finalize_scalar(&partial)?)?, 1.0);
        Ok(())
    }
}
