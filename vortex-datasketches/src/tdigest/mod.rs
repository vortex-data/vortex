// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod primitive;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use datasketches::tdigest::TDigestMut;
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

use self::primitive::accumulate_primitive;
use self::primitive::update_primitive;

/// Default t-digest compression parameter used by [`TDigestOptions`].
pub const DEFAULT_K: u16 = 200;

const OPTIONS_VERSION: u8 = 1;

/// Options for the `datasketches.tdigest` aggregate function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TDigestOptions {
    /// Apache DataSketches t-digest compression parameter. Must be at least `10`.
    pub k: u16,
}

impl Default for TDigestOptions {
    fn default() -> Self {
        Self { k: DEFAULT_K }
    }
}

impl Display for TDigestOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "k={}", self.k)
    }
}

impl TDigestOptions {
    fn validate(&self) -> VortexResult<()> {
        vortex_ensure!(
            self.k >= 10,
            "t-digest k must be at least 10, got {}",
            self.k
        );
        Ok(())
    }
}

/// Aggregate function vtable for Apache DataSketches t-digest sketches.
#[derive(Clone, Debug)]
pub struct TDigest;

/// Partial accumulator state for [`TDigest`].
pub struct TDigestPartial {
    k: u16,
    digest: TDigestMut,
}

impl TDigestPartial {
    fn update(&mut self, value: f64) {
        self.digest.update(value);
    }
}

/// Build a serialized Apache DataSketches t-digest sketch from a numeric array.
pub fn tdigest(
    array: &ArrayRef,
    options: TDigestOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(TDigest, options, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    acc.finish()
}

/// Return the approximate quantile from a serialized t-digest sketch.
pub fn quantile(bytes: &[u8], rank: f64) -> VortexResult<Option<f64>> {
    let digest = TDigestMut::deserialize(bytes, false)
        .map_err(|err| vortex_err!("Failed to deserialize t-digest sketch: {err}"))?;
    Ok(digest.freeze().quantile(rank))
}

impl AggregateFnVTable for TDigest {
    type Options = TDigestOptions;
    type Partial = TDigestPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("datasketches.tdigest")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        options.validate()?;
        let mut bytes = Vec::with_capacity(3);
        bytes.push(OPTIONS_VERSION);
        bytes.extend_from_slice(&options.k.to_le_bytes());
        Ok(Some(bytes))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_ensure!(
            metadata.len() == 3,
            "Invalid t-digest options metadata length: expected 3, got {}",
            metadata.len()
        );
        vortex_ensure!(
            metadata[0] == OPTIONS_VERSION,
            "Unsupported t-digest options metadata version: {}",
            metadata[0]
        );
        let options = TDigestOptions {
            k: u16::from_le_bytes([metadata[1], metadata[2]]),
        };
        options.validate()?;
        Ok(options)
    }

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        if options.validate().is_err() {
            return None;
        }
        match input_dtype {
            DType::Null | DType::Primitive(..) => Some(DType::Binary(Nullability::NonNullable)),
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
        Ok(TDigestPartial {
            k: options.k,
            digest: TDigestMut::try_new(options.k)
                .map_err(|err| vortex_err!("Failed to create t-digest: {err}"))?,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }

        let bytes = other
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("t-digest partial scalar must be non-null"))?;
        let other = TDigestMut::deserialize(bytes.as_slice(), false)
            .map_err(|err| vortex_err!("Failed to deserialize t-digest partial: {err}"))?;
        partial.digest.merge(&other);
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let mut digest = partial.digest.clone();
        Ok(Scalar::binary(digest.serialize(), Nullability::NonNullable))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.digest = TDigestMut::new(partial.k);
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
                if c.scalar().is_null() {
                    return Ok(());
                }
                match c.scalar().dtype() {
                    DType::Primitive(..) => {
                        for _ in 0..c.len() {
                            update_primitive(partial, c.scalar().as_primitive())?;
                        }
                        Ok(())
                    }
                    DType::Null => Ok(()),
                    _ => vortex_bail!(
                        "Unsupported constant dtype for t-digest: {}",
                        c.scalar().dtype()
                    ),
                }
            }
            Columnar::Canonical(c) => match c {
                Canonical::Null(_) => Ok(()),
                Canonical::Primitive(array) => accumulate_primitive(partial, array, ctx),
                Canonical::Bool(_)
                | Canonical::Decimal(_)
                | Canonical::Extension(_)
                | Canonical::VarBinView(_)
                | Canonical::Struct(_)
                | Canonical::List(_)
                | Canonical::FixedSizeList(_)
                | Canonical::Variant(_) => {
                    vortex_bail!("Unsupported canonical type for t-digest: {}", c.dtype())
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
    use datasketches::tdigest::TDigestMut;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;

    fn quantile_scalar(scalar: Scalar, rank: f64) -> VortexResult<Option<f64>> {
        let bytes = scalar
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("t-digest result must be non-null"))?;
        quantile(bytes.as_slice(), rank)
    }

    #[test]
    fn tdigest_skips_nulls() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();
        let array =
            PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(2.0), Some(3.0), None])
                .into_array();

        let result = tdigest(&array, TDigestOptions::default(), &mut ctx)?;

        assert_eq!(quantile_scalar(result, 0.0)?, Some(1.0));
        Ok(())
    }

    #[test]
    fn tdigest_combines_partial_sketches() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::Nullable);
        let mut partial = TDigest.empty_partial(&TDigestOptions::default(), &dtype)?;

        let mut left = TDigestMut::new(DEFAULT_K);
        left.update(1.0);
        left.update(2.0);
        TDigest.combine_partials(
            &mut partial,
            Scalar::binary(left.serialize(), Nullability::NonNullable),
        )?;

        let mut right = TDigestMut::new(DEFAULT_K);
        right.update(3.0);
        TDigest.combine_partials(
            &mut partial,
            Scalar::binary(right.serialize(), Nullability::NonNullable),
        )?;

        let result = TDigest.finalize_scalar(&partial)?;
        assert_eq!(quantile_scalar(result, 0.0)?, Some(1.0));
        Ok(())
    }
}
