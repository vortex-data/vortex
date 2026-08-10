// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Nullable execution strategies derived from a concrete row dispatch.

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::SinkResult;

/// The execution policy and output dtype selected by a planning visit.
pub struct BatchPlan {
    /// The non-nullable dtype built by the selected output capability.
    pub output_dtype: DType,

    /// How this concrete dispatch executes nullable rows.
    pub policy: RowPolicy,
}

impl BatchPlan {
    /// Return the output dtype widened with strict input nullability.
    pub fn result_dtype(&self, args: &[DType]) -> DType {
        let nullability = self.output_dtype.nullability()
            | Nullability::from(args.iter().any(DType::is_nullable));

        self.output_dtype.with_nullability(nullability)
    }
}

/// The nullable execution policy derived from one concrete dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowPolicy {
    /// Evaluate all rows and mask the result.
    Dense,

    /// Evaluate all rows, retrying only valid rows if a deferred error is raised.
    DenseWithRetry,

    /// Execute only valid rows, trying skip-invalid execution before filtering.
    ValidOnly,
}

impl RowPolicy {
    /// The policy for an infallible owned output.
    pub const fn for_owned_output<Args: ElementTuple>() -> Self {
        if Args::DENSE_SAFE && !Args::DECODE_FALLIBLE {
            Self::Dense
        } else {
            Self::ValidOnly
        }
    }

    /// The policy for an owned output carrying batch-deferred failure evidence.
    pub const fn for_deferred_output<Args: ElementTuple>() -> Self {
        if Args::DENSE_SAFE && !Args::DECODE_FALLIBLE {
            Self::DenseWithRetry
        } else {
            Self::ValidOnly
        }
    }

    /// The policy one concrete dispatch executes nullable rows under.
    ///
    /// Batch execution always tries [`reduce_encoded`](crate::scalar_fn::RowFn::reduce_encoded)
    /// against the original arrays before it tries the sink or filters the inputs. Skipping that
    /// probe can change the result of an encoding-aware function.
    pub const fn for_sink<Args: ElementTuple, ApplyResult: SinkResult>() -> Self {
        if Args::DENSE_SAFE && !Args::DECODE_FALLIBLE && !ApplyResult::FALLIBLE {
            Self::Dense
        } else {
            Self::ValidOnly
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::RowPolicy;
    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::dtype::DType;
    use crate::scalar_fn::InputElement;

    struct SparseFallibleElement;

    // SAFETY: the varying view reports length zero, so no index satisfies the unchecked-read
    // precondition.
    unsafe impl InputElement for SparseFallibleElement {
        type Column = ();
        type Varying<'a> = ();
        type Elem<'a> = ();

        const DENSE_SAFE: bool = false;
        const DECODE_FALLIBLE: bool = true;

        fn validate(_dtype: &DType) -> VortexResult<()> {
            Ok(())
        }

        fn decode(_array: ArrayRef, _ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
            Ok(())
        }

        fn get(_column: &Self::Column, _index: usize) -> Self::Elem<'_> {}

        fn varying(_column: &Self::Column) -> Self::Varying<'_> {}

        fn varying_len(_column: &Self::Varying<'_>) -> usize {
            0
        }

        fn get_varying<'a>(_column: &Self::Varying<'a>, _index: usize) -> Self::Elem<'a> {}
    }

    #[test]
    fn test_owned_output_policy() {
        assert_eq!(RowPolicy::for_owned_output::<(i64,)>(), RowPolicy::Dense);
        assert_eq!(
            RowPolicy::for_owned_output::<(SparseFallibleElement,)>(),
            RowPolicy::ValidOnly,
        );
    }

    #[test]
    fn test_deferred_output_policy() {
        assert_eq!(
            RowPolicy::for_deferred_output::<(i64,)>(),
            RowPolicy::DenseWithRetry,
        );
        assert_eq!(
            RowPolicy::for_deferred_output::<(SparseFallibleElement,)>(),
            RowPolicy::ValidOnly,
        );
    }

    #[test]
    fn test_sink_policy() {
        assert_eq!(RowPolicy::for_sink::<(i64,), ()>(), RowPolicy::Dense);
        assert_eq!(
            RowPolicy::for_sink::<(i64,), VortexResult<()>>(),
            RowPolicy::ValidOnly,
        );
        assert_eq!(
            RowPolicy::for_sink::<(SparseFallibleElement,), ()>(),
            RowPolicy::ValidOnly,
        );
    }
}
