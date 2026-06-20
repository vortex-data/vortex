// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::num::NonZeroUsize;
use std::sync::LazyLock;

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnSatisfaction;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::partial_ord::partial_max;
use crate::scalar::Scalar;
use crate::scalar::ScalarTruncation;
use crate::scalar::upper_bound;

/// Field name for the bounded maximum upper-bound value in the partial state.
pub const BOUNDED_MAX_BOUND: &str = "bound";
/// Field name for whether the partial state represents an unknown upper bound.
pub const BOUNDED_MAX_UNKNOWN: &str = "unknown";

static NAMES: LazyLock<FieldNames> =
    LazyLock::new(|| FieldNames::from([BOUNDED_MAX_BOUND, BOUNDED_MAX_UNKNOWN]));

/// Options for [`BoundedMax`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundedMaxOptions {
    /// Maximum byte length for UTF8/Binary bounds.
    pub max_bytes: NonZeroUsize,
}

impl Display for BoundedMaxOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.max_bytes.get())
    }
}

/// Compute a byte-bounded upper bound for the maximum non-null value of a UTF8/Binary array.
#[derive(Clone, Debug)]
pub struct BoundedMax;

enum BoundedMaxState {
    Empty,
    Value(Scalar),
    Unknown,
}

/// Partial accumulator state for the bounded maximum aggregate.
pub struct BoundedMaxPartial {
    state: BoundedMaxState,
    element_dtype: DType,
    max_bytes: NonZeroUsize,
}

impl BoundedMaxPartial {
    fn merge_bound(&mut self, max: Scalar) {
        if max.is_null() {
            return;
        }

        self.state = match std::mem::replace(&mut self.state, BoundedMaxState::Empty) {
            BoundedMaxState::Empty => BoundedMaxState::Value(max),
            BoundedMaxState::Value(current) => BoundedMaxState::Value(
                partial_max(max, current).vortex_expect("incomparable bounded max scalars"),
            ),
            BoundedMaxState::Unknown => BoundedMaxState::Unknown,
        };
    }

    fn unknown(&mut self) {
        self.state = BoundedMaxState::Unknown;
    }

    fn final_scalar(&self) -> VortexResult<Scalar> {
        let dtype = self.element_dtype.as_nullable();
        match &self.state {
            BoundedMaxState::Value(max) => max.cast(&dtype),
            BoundedMaxState::Empty | BoundedMaxState::Unknown => Ok(Scalar::null(dtype)),
        }
    }
}

/// Return the serialized partial-state dtype for [`BoundedMax`].
///
/// A null struct means the partial is empty. A non-null struct with a null `bound` and
/// `unknown = true` means the input has a non-null maximum but no finite upper bound could be
/// represented within the configured byte limit.
pub fn make_bounded_max_partial_dtype(element_dtype: &DType) -> DType {
    DType::Struct(
        StructFields::new(
            NAMES.clone(),
            vec![
                element_dtype.as_nullable(),
                DType::Bool(Nullability::NonNullable),
            ],
        ),
        Nullability::Nullable,
    )
}

impl AggregateFnVTable for BoundedMax {
    type Options = BoundedMaxOptions;
    type Partial = BoundedMaxPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.bounded_max")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let max_bytes = u64::try_from(options.max_bytes.get())?;
        Ok(Some(max_bytes.to_le_bytes().to_vec()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_ensure!(
            metadata.len() == size_of::<u64>(),
            "BoundedMax options expected {} bytes, got {}",
            size_of::<u64>(),
            metadata.len()
        );
        let mut bytes = [0u8; size_of::<u64>()];
        bytes.copy_from_slice(metadata);
        let max_bytes = usize::try_from(u64::from_le_bytes(bytes))?;
        vortex_ensure!(max_bytes > 0, "BoundedMax requires max_bytes > 0");
        Ok(BoundedMaxOptions {
            max_bytes: NonZeroUsize::new(max_bytes).vortex_expect("checked non-zero max_bytes"),
        })
    }

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        supported_dtype(options, input_dtype).map(DType::as_nullable)
    }

    fn can_satisfy(
        &self,
        options: &Self::Options,
        requested: &AggregateFnRef,
    ) -> AggregateFnSatisfaction {
        if let Some(other) = requested.as_opt::<Self>() {
            return if other == options {
                AggregateFnSatisfaction::Exact
            } else if options.max_bytes >= other.max_bytes {
                AggregateFnSatisfaction::Approximate
            } else {
                AggregateFnSatisfaction::No
            };
        }

        if requested.is::<Max>() {
            AggregateFnSatisfaction::Approximate
        } else {
            AggregateFnSatisfaction::No
        }
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        supported_dtype(options, input_dtype).map(make_bounded_max_partial_dtype)
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(BoundedMaxPartial {
            state: BoundedMaxState::Empty,
            element_dtype: input_dtype.clone(),
            max_bytes: options.max_bytes,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }

        let Some(other) = other.as_struct_opt() else {
            vortex_bail!("BoundedMax partial must be a struct, got {}", other.dtype());
        };
        let Some(bound) = other.field_by_idx(0) else {
            vortex_bail!("BoundedMax partial is missing its bound field");
        };
        let Some(unknown) = other
            .field_by_idx(1)
            .and_then(|unknown| unknown.as_bool().value())
        else {
            vortex_bail!("BoundedMax partial is missing its non-null unknown field");
        };

        if unknown {
            partial.unknown();
        } else {
            partial.merge_bound(bound);
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let dtype = make_bounded_max_partial_dtype(&partial.element_dtype);
        let bound_dtype = partial.element_dtype.as_nullable();
        match &partial.state {
            BoundedMaxState::Empty => Ok(Scalar::null(dtype)),
            BoundedMaxState::Value(max) => Ok(Scalar::struct_(
                dtype,
                vec![
                    max.cast(&bound_dtype)?,
                    Scalar::bool(false, Nullability::NonNullable),
                ],
            )),
            BoundedMaxState::Unknown => Ok(Scalar::struct_(
                dtype,
                vec![
                    Scalar::null(bound_dtype),
                    Scalar::bool(true, Nullability::NonNullable),
                ],
            )),
        }
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.state = BoundedMaxState::Empty;
    }

    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        matches!(partial.state, BoundedMaxState::Unknown)
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Delegate to the existing min_max implementation for now. A dedicated bounded-max
        // aggregate would avoid computing min when only max is needed.
        let array = match batch {
            Columnar::Canonical(canonical) => canonical.clone().into_array(),
            Columnar::Constant(constant) => constant.clone().into_array(),
        };
        let Some(result) = min_max(&array, ctx)? else {
            return Ok(());
        };
        match truncate_max(result.max, partial.max_bytes.get())? {
            Some(bound) => partial.merge_bound(bound),
            None => partial.unknown(),
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        partials.get_item(BOUNDED_MAX_BOUND)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        partial.final_scalar()
    }
}

fn supported_dtype<'a>(_options: &BoundedMaxOptions, input_dtype: &'a DType) -> Option<&'a DType> {
    MinMax
        .return_dtype(&EmptyOptions, input_dtype)
        .map(|_| input_dtype)
}

fn truncate_max(value: Scalar, max_bytes: usize) -> VortexResult<Option<Scalar>> {
    let nullability = value.dtype().nullability();
    match value.dtype() {
        DType::Utf8(_) => {
            Ok(
                upper_bound(BufferString::from_scalar(value)?, max_bytes, nullability)
                    .map(|(bound, _)| bound),
            )
        }
        DType::Binary(_) => {
            Ok(
                upper_bound(ByteBuffer::from_scalar(value)?, max_bytes, nullability)
                    .map(|(bound, _)| bound),
            )
        }
        _ => Ok(Some(value)),
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnSatisfaction;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::bounded_max::BoundedMax;
    use crate::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
    use crate::aggregate_fn::fns::bounded_max::make_bounded_max_partial_dtype;
    use crate::aggregate_fn::fns::max::Max;
    use crate::aggregate_fn::fns::min::Min;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    fn max_bytes(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).vortex_expect("non-zero max_bytes")
    }

    fn fresh_session() -> VortexSession {
        crate::array_session()
    }

    #[test]
    fn bounded_max_truncates_utf8_to_upper_bound() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = VarBinViewArray::from_iter_str(["aardvark", "char🪩"]).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMax,
            BoundedMaxOptions {
                max_bytes: max_bytes(5),
            },
            array.dtype().clone(),
        )?;

        acc.accumulate(&array, &mut ctx)?;

        assert_eq!(acc.finish()?, Scalar::utf8("chas", Nullability::Nullable));
        Ok(())
    }

    #[test]
    fn bounded_max_unknown_upper_bound_returns_null() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = VarBinViewArray::from_iter_bin([&[255u8, 255, 255][..]]).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMax,
            BoundedMaxOptions {
                max_bytes: max_bytes(2),
            },
            array.dtype().clone(),
        )?;

        acc.accumulate(&array, &mut ctx)?;

        assert_eq!(acc.finish()?, Scalar::null(array.dtype().as_nullable()));
        Ok(())
    }

    #[test]
    fn bounded_max_empty_does_not_poison_later_values() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let empty = VarBinViewArray::from_iter_bin(Vec::<&[u8]>::new()).into_array();
        let values = VarBinViewArray::from_iter_bin([&[1u8][..]]).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMax,
            BoundedMaxOptions {
                max_bytes: max_bytes(2),
            },
            empty.dtype().clone(),
        )?;

        acc.accumulate(&empty, &mut ctx)?;
        acc.accumulate(&values, &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::binary(buffer![1u8], Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn bounded_max_unknown_poisons_later_values() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let unknown = VarBinViewArray::from_iter_bin([&[255u8, 255, 255][..]]).into_array();
        let values = VarBinViewArray::from_iter_bin([&[1u8][..]]).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMax,
            BoundedMaxOptions {
                max_bytes: max_bytes(2),
            },
            unknown.dtype().clone(),
        )?;

        acc.accumulate(&unknown, &mut ctx)?;
        acc.accumulate(&values, &mut ctx)?;

        assert_eq!(acc.finish()?, Scalar::null(unknown.dtype().as_nullable()));
        Ok(())
    }

    #[test]
    fn bounded_max_empty_partial_does_not_poison_existing_bound() -> VortexResult<()> {
        let mut ctx = fresh_session().create_execution_ctx();
        let values = VarBinViewArray::from_iter_bin([&[1u8][..]]).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMax,
            BoundedMaxOptions {
                max_bytes: max_bytes(2),
            },
            values.dtype().clone(),
        )?;

        acc.accumulate(&values, &mut ctx)?;
        acc.combine_partials(Scalar::null(make_bounded_max_partial_dtype(values.dtype())))?;

        assert_eq!(
            acc.finish()?,
            Scalar::binary(buffer![1u8], Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn bounded_max_unknown_partial_poisons_existing_bound() -> VortexResult<()> {
        let mut ctx = fresh_session().create_execution_ctx();
        let values = VarBinViewArray::from_iter_bin([&[1u8][..]]).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMax,
            BoundedMaxOptions {
                max_bytes: max_bytes(2),
            },
            values.dtype().clone(),
        )?;

        let partial_dtype = make_bounded_max_partial_dtype(values.dtype());
        let unknown = Scalar::struct_(
            partial_dtype,
            vec![
                Scalar::null(values.dtype().as_nullable()),
                Scalar::bool(true, Nullability::NonNullable),
            ],
        );

        acc.accumulate(&values, &mut ctx)?;
        acc.combine_partials(unknown)?;

        assert_eq!(acc.finish()?, Scalar::null(values.dtype().as_nullable()));
        Ok(())
    }

    #[test]
    fn bounded_max_keeps_fixed_width_values_exact() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = PrimitiveArray::new(buffer![10i32, 20, 5], Validity::NonNullable).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMax,
            BoundedMaxOptions {
                max_bytes: max_bytes(9),
            },
            array.dtype().clone(),
        )?;

        acc.accumulate(&array, &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::primitive(20i32, Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn bounded_max_satisfies_max_bounds() {
        let stored = BoundedMax.bind(BoundedMaxOptions {
            max_bytes: max_bytes(5),
        });
        let same = BoundedMax.bind(BoundedMaxOptions {
            max_bytes: max_bytes(5),
        });
        let looser_bounded = BoundedMax.bind(BoundedMaxOptions {
            max_bytes: max_bytes(4),
        });
        let tighter_bounded = BoundedMax.bind(BoundedMaxOptions {
            max_bytes: max_bytes(6),
        });

        assert_eq!(stored.can_satisfy(&same), AggregateFnSatisfaction::Exact);
        assert_eq!(
            stored.can_satisfy(&looser_bounded),
            AggregateFnSatisfaction::Approximate
        );
        assert_eq!(
            stored.can_satisfy(&tighter_bounded),
            AggregateFnSatisfaction::No
        );
        assert_eq!(
            stored.can_satisfy(&Max.bind(EmptyOptions)),
            AggregateFnSatisfaction::Approximate
        );
        assert_eq!(
            Max.bind(EmptyOptions).can_satisfy(&stored),
            AggregateFnSatisfaction::Approximate
        );
        assert_eq!(
            stored.can_satisfy(&Min.bind(EmptyOptions)),
            AggregateFnSatisfaction::No
        );
    }

    #[test]
    fn bounded_max_options_round_trip() -> VortexResult<()> {
        let options = BoundedMaxOptions {
            max_bytes: max_bytes(64),
        };
        let metadata = BoundedMax
            .serialize(&options)?
            .expect("serializable options");
        let roundtrip = BoundedMax.deserialize(&metadata, &VortexSession::empty())?;

        assert_eq!(roundtrip, options);
        Ok(())
    }
}
