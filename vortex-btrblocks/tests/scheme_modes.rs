// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VTable;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_btrblocks::ArrayAndStats;
    use vortex_btrblocks::BtrBlocksCompressorBuilder;
    use vortex_btrblocks::CascadingCompressor;
    use vortex_btrblocks::CompressorContext;
    use vortex_btrblocks::Scheme;
    use vortex_btrblocks::SchemeExt;
    use vortex_btrblocks::schemes::integer::BitPackingScheme;
    use vortex_compressor::scheme::CompressionEstimate;
    use vortex_compressor::scheme::EstimateVerdict;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPacked;
    use vortex_fastlanes::Delta;
    use vortex_session::registry::CachedId;

    static NEWER_ID: CachedId = CachedId::new("test.delta_newer");

    /// A scheme with two wire formats. It always writes `Delta`, and when the writer permits the
    /// newer format it takes a different branch. The newer branch stands in for a format this test
    /// cannot serialize, so it returns the input unchanged and the compressor falls back to
    /// canonical output.
    #[derive(Debug)]
    struct TwoFormatDelta;

    impl Scheme for TwoFormatDelta {
        fn scheme_name(&self) -> &'static str {
            "test.two_format_delta"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            canonical.dtype().is_int()
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            vec![Delta.id()]
        }

        /// Children: bases=0, deltas=1.
        fn num_children(&self) -> usize {
            2
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
        }

        fn compress(
            &self,
            compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            if compress_ctx.allows_serialized_id(&NEWER_ID) {
                return Ok(data.array().clone());
            }
            let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
            let (bases, deltas) = vortex_fastlanes::delta_compress(&primitive, exec_ctx)?;
            let bases = compressor.compress_child(
                &bases.into_array(),
                &compress_ctx,
                self.id(),
                0,
                exec_ctx,
            )?;
            let deltas = compressor.compress_child(
                &deltas.into_array(),
                &compress_ctx,
                self.id(),
                1,
                exec_ctx,
            )?;
            Delta::try_new(bases, deltas, 0, primitive.len()).map(IntoArray::into_array)
        }
    }

    fn compressor(allowed: &[ArrayId]) -> vortex_btrblocks::BtrBlocksCompressor {
        BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&TwoFormatDelta)
            .with_new_scheme(&BitPackingScheme)
            .retain_allowed_encodings(&allowed.iter().copied().collect())
            .build()
    }

    /// The scheme reads the writer's permitted IDs from its context and picks its format.
    #[rstest]
    #[case::frozen_only(vec![Delta.id(), BitPacked.id()], true)]
    #[case::newer_permitted(vec![Delta.id(), BitPacked.id(), *NEWER_ID], false)]
    fn a_scheme_picks_its_format_from_the_permitted_ids(
        #[case] allowed: Vec<ArrayId>,
        #[case] expect_delta: bool,
    ) -> VortexResult<()> {
        let session = array_session();
        vortex_fastlanes::initialize(&session);
        let compressor = compressor(&allowed);
        let array = PrimitiveArray::from_iter((0..65_536u32).map(|i| i / 3)).into_array();
        let mut ctx = session.create_execution_ctx();
        let compressed = compressor.compress(&array, &mut ctx)?;
        assert_eq!(compressed.encoding_id() == Delta.id(), expect_delta);
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    /// The declared format is required. Permitting only the newer one leaves the scheme out.
    #[test]
    fn the_declared_format_must_be_permitted() {
        let compressor = compressor(&[*NEWER_ID, BitPacked.id()]);
        assert!(!compressor.has_scheme(TwoFormatDelta.id()));
        assert!(compressor.has_scheme(BitPackingScheme.id()));
    }
}
