// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression plans: the scheme decisions made while compressing one array, which can be replayed
//! as a hint when compressing a similar array (e.g. the next chunk of the same column).

use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::scheme::SchemeId;

/// Location of a scheme decision within a compression tree.
///
/// Each step is the `(scheme_id, child_index)` that was descended through, including structural
/// steps (struct fields, list elements, ...) that do not consume cascade depth.
pub(crate) type PlanPath = Vec<(SchemeId, usize)>;

/// A scheme decision recorded at a single [`PlanPath`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PlanDecision {
    /// No scheme beat the canonical encoding.
    Uncompressed,
    /// The scheme was applied and achieved the given compression ratio.
    Scheme {
        /// The scheme that was applied.
        id: SchemeId,
        /// The achieved ratio of uncompressed to compressed bytes.
        ratio: f64,
    },
}

/// The scheme decisions made while compressing an array.
///
/// Obtained from [`CascadingCompressor::compress_with_plan`] and passed back as a hint when
/// compressing a similar array. Replaying a plan skips scheme selection (sampling) wherever the
/// hinted scheme is still applicable and compresses at least nearly as well as it did when the
/// plan was recorded. Elsewhere the compressor falls back to a full search.
///
/// [`CascadingCompressor::compress_with_plan`]: crate::CascadingCompressor::compress_with_plan
#[derive(Debug, Clone, Default)]
pub struct CompressionPlan {
    /// Decisions keyed by their location in the compression tree.
    decisions: FxHashMap<PlanPath, PlanDecision>,
}

impl CompressionPlan {
    /// Returns the number of recorded decisions.
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Returns `true` if the plan has no recorded decisions.
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    /// Returns the decision recorded at `path`, if any.
    pub(crate) fn get(&self, path: &[(SchemeId, usize)]) -> Option<PlanDecision> {
        self.decisions.get(path).copied()
    }
}

/// Plan state shared by every [`CompressorContext`](crate::scheme::CompressorContext) of a single
/// compression call.
#[derive(Debug, Default)]
pub(crate) struct PlanState {
    /// The plan to replay, if any.
    hint: Option<Arc<CompressionPlan>>,
    /// Decisions made so far, in order. Later entries for the same path win.
    log: Mutex<Vec<(PlanPath, PlanDecision)>>,
}

impl PlanState {
    /// Creates plan state replaying `hint`.
    pub(crate) fn new(hint: Option<Arc<CompressionPlan>>) -> Self {
        Self {
            hint,
            log: Mutex::default(),
        }
    }

    /// Returns the hinted decision at `path`, if any.
    pub(crate) fn hint(&self, path: &[(SchemeId, usize)]) -> Option<PlanDecision> {
        self.hint.as_ref().and_then(|plan| plan.get(path))
    }

    /// Records a decision at `path`.
    pub(crate) fn record(&self, path: &[(SchemeId, usize)], decision: PlanDecision) {
        self.log.lock().push((path.to_vec(), decision));
    }

    /// Returns a marker that [`rollback`](Self::rollback) can rewind the log to.
    pub(crate) fn checkpoint(&self) -> usize {
        self.log.lock().len()
    }

    /// Discards every decision recorded after `checkpoint`.
    pub(crate) fn rollback(&self, checkpoint: usize) {
        self.log.lock().truncate(checkpoint);
    }

    /// Builds the plan from the recorded decisions.
    pub(crate) fn finish(&self) -> CompressionPlan {
        CompressionPlan {
            decisions: self.log.lock().drain(..).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::ArrayId;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::CascadingCompressor;
    use crate::builtins::IntDictScheme;
    use crate::scheme::CompressionEstimate;
    use crate::scheme::CompressorContext;
    use crate::scheme::DeferredEstimate;
    use crate::scheme::Scheme;
    use crate::stats::ArrayAndStats;
    use crate::stats::GenerateStatsOptions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    thread_local! {
        static SAMPLE_COMPRESSIONS: Cell<usize> = const { Cell::new(0) };
    }

    /// Dictionary encodes integers (without compressing the dictionary's children), counting how
    /// often it compresses a sample on this thread.
    #[derive(Debug)]
    struct CountingDictScheme;

    impl Scheme for CountingDictScheme {
        fn scheme_name(&self) -> &'static str {
            "test.counting_dict"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches!(canonical, Canonical::Primitive(primitive) if primitive.ptype().is_int())
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            IntDictScheme.produced_encodings()
        }

        fn stats_options(&self) -> GenerateStatsOptions {
            IntDictScheme.stats_options()
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Deferred(DeferredEstimate::Sample)
        }

        fn compress(
            &self,
            compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            if compress_ctx.is_sample() {
                SAMPLE_COMPRESSIONS.with(|count| count.set(count.get() + 1));
            }
            IntDictScheme.compress(compressor, data, compress_ctx.as_leaf(), exec_ctx)
        }
    }

    fn compressor() -> CascadingCompressor {
        CascadingCompressor::new(vec![&CountingDictScheme])
    }

    fn take_sample_compressions() -> usize {
        SAMPLE_COMPRESSIONS.with(|count| count.replace(0))
    }

    /// Integers cycling through `cardinality` distinct values.
    fn ints(cardinality: i32, offset: i32) -> ArrayRef {
        PrimitiveArray::new(
            Buffer::from_iter((0..8192).map(|i| i % cardinality + offset)),
            Validity::NonNullable,
        )
        .into_array()
    }

    fn encodings(array: &ArrayRef) -> String {
        array.display_tree_encodings_only().to_string()
    }

    #[test]
    fn replay_skips_sampling() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressor = compressor();

        let first = ints(16, 0);
        let (first_compressed, plan) = compressor.compress_with_plan(&first, None, &mut ctx)?;
        assert!(take_sample_compressions() > 0);
        assert_eq!(plan.len(), 1);

        let second = ints(16, 100);
        let (second_compressed, replayed_plan) =
            compressor.compress_with_plan(&second, Some(Arc::new(plan)), &mut ctx)?;
        assert_eq!(take_sample_compressions(), 0);
        assert_eq!(encodings(&second_compressed), encodings(&first_compressed));
        assert_arrays_eq!(second_compressed, second, &mut ctx);
        assert_eq!(replayed_plan.len(), 1);
        Ok(())
    }

    #[test]
    fn replay_falls_back_to_search_when_data_drifts() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressor = compressor();

        let (_, plan) = compressor.compress_with_plan(&ints(4, 0), None, &mut ctx)?;
        take_sample_compressions();

        // Every value is distinct, so the replayed dictionary is far worse than recorded.
        let drifted = ints(8192, 0);
        let (compressed, drifted_plan) =
            compressor.compress_with_plan(&drifted, Some(Arc::new(plan)), &mut ctx)?;
        assert!(take_sample_compressions() > 0);
        assert_eq!(encodings(&compressed), encodings(&compressor.compress(&drifted, &mut ctx)?));
        assert_arrays_eq!(compressed, drifted, &mut ctx);
        assert_eq!(drifted_plan.len(), 1);
        Ok(())
    }

    #[test]
    fn struct_fields_have_separate_decisions() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressor = compressor();
        let array = |offset| -> VortexResult<ArrayRef> {
            Ok(StructArray::from_fields(&[
                ("dict", ints(16, offset)),
                ("distinct", ints(8192, offset)),
            ])?
            .into_array())
        };

        let first = array(0)?;
        let (first_compressed, plan) = compressor.compress_with_plan(&first, None, &mut ctx)?;
        assert_eq!(plan.len(), 2);
        take_sample_compressions();

        let second = array(100)?;
        let (second_compressed, _) =
            compressor.compress_with_plan(&second, Some(Arc::new(plan)), &mut ctx)?;
        assert_eq!(take_sample_compressions(), 0);
        assert_eq!(encodings(&second_compressed), encodings(&first_compressed));
        assert_arrays_eq!(second_compressed, second, &mut ctx);
        Ok(())
    }
}
