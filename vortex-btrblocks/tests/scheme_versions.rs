// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(feature = "unstable_encodings")]

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
    use vortex_btrblocks::CascadingCompressor;
    use vortex_btrblocks::CompressorContext;
    use vortex_btrblocks::Scheme;
    use vortex_btrblocks::SchemeExt;
    use vortex_btrblocks::schemes::integer::DeltaScheme;
    use vortex_btrblocks::schemes::integer::IntRLEScheme;
    use vortex_compressor::scheme::CompressionEstimate;
    use vortex_compressor::scheme::EstimateVerdict;
    use vortex_error::VortexResult;
    use vortex_fastlanes::Delta;
    use vortex_fastlanes::RLE;
    use vortex_session::registry::CachedId;

    static DELTA_V2_ID: CachedId = CachedId::new("test.delta_v2");
    static DELTA_V1: DeltaScheme = DeltaScheme::new(1.25);

    #[derive(Debug)]
    struct DeltaV2;

    impl Scheme for DeltaV2 {
        fn scheme_name(&self) -> &'static str {
            "test.delta_v2"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            DELTA_V1.matches(canonical)
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            vec![*DELTA_V2_ID]
        }

        fn predecessor(&self) -> Option<&'static dyn Scheme> {
            Some(&DELTA_V1)
        }

        fn num_children(&self) -> usize {
            2
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            Ok(data.array().clone())
        }
    }

    #[rstest]
    #[case::predecessor(Delta.id(), true)]
    #[case::replacement(*DELTA_V2_ID, false)]
    fn rle_respects_selected_delta_version(
        #[case] allowed_delta: ArrayId,
        #[case] expect_delta: bool,
    ) -> VortexResult<()> {
        let session = array_session();
        vortex_fastlanes::initialize(&session);
        let compressor = CascadingCompressor::new(vec![&IntRLEScheme, &DeltaV2])
            .with_allowed_serialized_ids([RLE.id(), allowed_delta].into_iter().collect());
        assert!(compressor.has_scheme_family(DELTA_V1.id()));
        let array = PrimitiveArray::from_iter((0..65_536u32).map(|i| (i / 64) % 100)).into_array();
        let mut ctx = session.create_execution_ctx();
        let compressed = compressor.compress(&array, &mut ctx)?;
        assert_eq!(compressed.encoding_id(), RLE.id());
        let has_delta = compressed
            .depth_first_traversal()
            .any(|array| array.encoding_id() == Delta.id());
        assert_eq!(has_delta, expect_delta);
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }
}
