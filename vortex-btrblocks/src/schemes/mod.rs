// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression scheme implementations.

pub mod binary;
pub mod boolean;
pub mod float;
pub mod integer;
pub mod string;

pub mod decimal;
pub mod fixed_size_list;
pub mod temporal;

pub(crate) mod patches;

use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::DecimalDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::SchemeExt;

use crate::schemes::integer::RunEndScheme;
use crate::schemes::integer::SparseScheme;

/// Shared descendant exclusion rules for RLE schemes.
///
/// RLE indices (child 1) and offsets (child 2) are monotonically increasing positions with all
/// unique values. Dict and Sparse are pointless on such data. Self-exclusion already prevents
/// RLE on RLE children.
fn rle_descendant_exclusions() -> Vec<DescendantExclusion> {
    vec![
        DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::Many(&[1, 2]),
        },
        // RLE indices count up within each chunk, so Delta packs them into a few bits. RunEnd
        // over them would run-length encode already run-length encoded data, which every RLE
        // decode and filter then has to expand twice.
        DescendantExclusion {
            excluded: RunEndScheme.id(),
            children: ChildSelection::One(1),
        },
        DescendantExclusion {
            excluded: SparseScheme.id(),
            children: ChildSelection::Many(&[1, 2]),
        },
    ]
}

/// Shared ancestor exclusion rules for RLE schemes.
///
/// Dict values (child 0) are all unique by definition, so RLE is pointless on them.
fn rle_ancestor_exclusions() -> Vec<AncestorExclusion> {
    vec![
        AncestorExclusion {
            ancestor: IntDictScheme.id(),
            children: ChildSelection::One(0),
        },
        AncestorExclusion {
            ancestor: FloatDictScheme.id(),
            children: ChildSelection::One(0),
        },
        AncestorExclusion {
            ancestor: StringDictScheme.id(),
            children: ChildSelection::One(0),
        },
        AncestorExclusion {
            ancestor: BinaryDictScheme.id(),
            children: ChildSelection::One(0),
        },
        AncestorExclusion {
            ancestor: DecimalDictScheme.id(),
            children: ChildSelection::One(0),
        },
    ]
}

#[cfg(all(test, feature = "pco"))]
pub(crate) mod test_utils {
    use vortex_array::ArrayId;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_compressor::CascadingCompressor;
    use vortex_compressor::scheme::CompressionEstimate;
    use vortex_error::VortexResult;
    use vortex_pco::Pco;

    use crate::ArrayAndStats;
    use crate::CompressorContext;
    use crate::Encoded;
    use crate::Scheme;

    /// Delegates every [`Scheme`] method to `S` except `fixed_nbytes`, which stays at the default `0`.
    #[derive(Debug, Copy, Clone)]
    pub(crate) struct UnamortizedScheme<S>(pub(crate) S);

    impl<S: Scheme> Scheme for UnamortizedScheme<S> {
        fn scheme_name(&self) -> &'static str {
            self.0.scheme_name()
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            self.0.matches(canonical)
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            self.0.produced_encodings()
        }

        fn expected_compression_ratio(
            &self,
            data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            self.0
                .expected_compression_ratio(data, compress_ctx, exec_ctx)
        }

        fn encode(
            &self,
            data: &ArrayAndStats,
            compress_ctx: &CompressorContext,
            exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<Encoded> {
            self.0.encode(data, compress_ctx, exec_ctx)
        }

        fn compress(
            &self,
            compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            self.0.compress(compressor, data, compress_ctx, exec_ctx)
        }
    }

    pub(crate) fn assert_pco_fixed_nbytes_and_fallback(
        scheme: &dyn Scheme,
        primitive: PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let pco = Pco::from_primitive(
            primitive.as_view(),
            pco::DEFAULT_COMPRESSION_LEVEL,
            8192,
            ctx,
        )?;
        let expected_fixed = pco.chunk_metas_nbytes();
        assert!(expected_fixed > 0);
        assert_eq!(scheme.fixed_nbytes(&pco.into_array()), expected_fixed);
        assert_eq!(scheme.fixed_nbytes(&primitive.into_array()), 0);
        Ok(())
    }
}
