// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::dtype::DType;
use crate::partial_ord::partial_min;
use crate::scalar::Scalar;
use crate::scalar::ScalarTruncation;
use crate::scalar::lower_bound;

/// Options for [`BoundedMin`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundedMinOptions {
    /// Maximum byte length for UTF8/Binary bounds.
    pub max_bytes: usize,
}

impl Display for BoundedMinOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.max_bytes)
    }
}

/// Compute a byte-bounded lower bound for the minimum non-null value of a UTF8/Binary array.
#[derive(Clone, Debug)]
pub struct BoundedMin;

/// Partial accumulator state for the bounded minimum aggregate.
pub struct BoundedMinPartial {
    min: Option<Scalar>,
    element_dtype: DType,
    max_bytes: usize,
}

impl BoundedMinPartial {
    fn merge(&mut self, min: Scalar) {
        if min.is_null() {
            return;
        }

        self.min = Some(match self.min.take() {
            Some(current) => {
                partial_min(min, current).vortex_expect("incomparable bounded min scalars")
            }
            None => min,
        });
    }
}

impl AggregateFnVTable for BoundedMin {
    type Options = BoundedMinOptions;
    type Partial = BoundedMinPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.bounded_min")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let max_bytes = u64::try_from(options.max_bytes)?;
        Ok(Some(max_bytes.to_le_bytes().to_vec()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_ensure!(
            metadata.len() == size_of::<u64>(),
            "BoundedMin options expected {} bytes, got {}",
            size_of::<u64>(),
            metadata.len()
        );
        let mut bytes = [0u8; size_of::<u64>()];
        bytes.copy_from_slice(metadata);
        let max_bytes = usize::try_from(u64::from_le_bytes(bytes))?;
        vortex_ensure!(max_bytes > 0, "BoundedMin requires max_bytes > 0");
        Ok(BoundedMinOptions { max_bytes })
    }

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        supported_dtype(options, input_dtype).map(DType::as_nullable)
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(BoundedMinPartial {
            min: None,
            element_dtype: input_dtype.clone(),
            max_bytes: options.max_bytes,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        partial.merge(other);
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let dtype = partial.element_dtype.as_nullable();
        match &partial.min {
            Some(min) => min.cast(&dtype),
            None => Ok(Scalar::null(dtype)),
        }
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.min = None;
    }

    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        false
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let array = match batch {
            Columnar::Canonical(canonical) => canonical.clone().into_array(),
            Columnar::Constant(constant) => constant.clone().into_array(),
        };
        let Some(result) = min_max(&array, ctx)? else {
            return Ok(());
        };
        if let Some(bound) = truncate_min(result.min, partial.max_bytes)? {
            partial.merge(bound);
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

fn supported_dtype<'a>(options: &BoundedMinOptions, input_dtype: &'a DType) -> Option<&'a DType> {
    if options.max_bytes == 0 {
        return None;
    }

    MinMax
        .return_dtype(&EmptyOptions, input_dtype)
        .map(|_| input_dtype)
}

fn truncate_min(value: Scalar, max_bytes: usize) -> VortexResult<Option<Scalar>> {
    let nullability = value.dtype().nullability();
    match value.dtype() {
        DType::Utf8(_) => {
            Ok(
                lower_bound(BufferString::from_scalar(value)?, max_bytes, nullability)
                    .map(|(bound, _)| bound),
            )
        }
        DType::Binary(_) => {
            Ok(
                lower_bound(ByteBuffer::from_scalar(value)?, max_bytes, nullability)
                    .map(|(bound, _)| bound),
            )
        }
        _ => Ok(Some(value)),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::fns::bounded_min::BoundedMin;
    use crate::aggregate_fn::fns::bounded_min::BoundedMinOptions;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn bounded_min_truncates_utf8_to_lower_bound() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array =
            VarBinViewArray::from_iter_str(["snowman⛄️snowman", "untruncated"]).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMin,
            BoundedMinOptions { max_bytes: 9 },
            array.dtype().clone(),
        )?;

        acc.accumulate(&array, &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::utf8("snowman", Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn bounded_min_keeps_fixed_width_values_exact() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = PrimitiveArray::new(buffer![10i32, 20, 5], Validity::NonNullable).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMin,
            BoundedMinOptions { max_bytes: 9 },
            array.dtype().clone(),
        )?;

        acc.accumulate(&array, &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::primitive(5i32, Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn bounded_min_options_round_trip() -> VortexResult<()> {
        let options = BoundedMinOptions { max_bytes: 64 };
        let metadata = BoundedMin
            .serialize(&options)?
            .expect("serializable options");
        let roundtrip = BoundedMin.deserialize(&metadata, &VortexSession::empty())?;

        assert_eq!(roundtrip, options);
        Ok(())
    }
}
