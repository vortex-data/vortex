// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_alp::{ALPEncoding, ALPRDEncoding};
use vortex_array::{ArrayRegistry, EncodingRef};
use vortex_bytebool::ByteBoolEncoding;
use vortex_datetime_parts::DateTimePartsEncoding;
use vortex_decimal_byte_parts::DecimalBytePartsEncoding;
use vortex_dict::DictEncoding;
use vortex_expr::{ExprRegistry, ExprRegistryExt};
use vortex_fastlanes::{BitPackedEncoding, DeltaEncoding, FoREncoding, RLEEncoding};
use vortex_fsst::FSSTEncoding;
use vortex_layout::{LayoutRegistry, LayoutRegistryExt};
use vortex_metrics::VortexMetrics;
use vortex_pco::PcoEncoding;
use vortex_runend::RunEndEncoding;
use vortex_sequence::SequenceEncoding;
use vortex_sparse::SparseEncoding;
use vortex_zigzag::ZigZagEncoding;

/// A Vortex session encapsulates the set of extensible arrays, layouts, compute functions, dtypes,
/// etc. that are available for use in a given context.
///
/// It is also the entry-point passed to dynamic libraries to initialize Vortex plugins.
#[derive(Debug)]
pub struct VortexSession {
    arrays: ArrayRegistry,
    layouts: LayoutRegistry,
    expressions: ExprRegistry,
    metrics: VortexMetrics,
}

impl Default for VortexSession {
    fn default() -> Self {
        // Register the compressed encodings that Vortex ships with.
        let mut arrays = ArrayRegistry::canonical_only();
        arrays.register_many([
            EncodingRef::new_ref(ALPEncoding.as_ref()),
            EncodingRef::new_ref(ALPRDEncoding.as_ref()),
            EncodingRef::new_ref(BitPackedEncoding.as_ref()),
            EncodingRef::new_ref(ByteBoolEncoding.as_ref()),
            EncodingRef::new_ref(DateTimePartsEncoding.as_ref()),
            EncodingRef::new_ref(DecimalBytePartsEncoding.as_ref()),
            EncodingRef::new_ref(DeltaEncoding.as_ref()),
            EncodingRef::new_ref(DictEncoding.as_ref()),
            EncodingRef::new_ref(FSSTEncoding.as_ref()),
            EncodingRef::new_ref(FoREncoding.as_ref()),
            EncodingRef::new_ref(PcoEncoding.as_ref()),
            EncodingRef::new_ref(RLEEncoding.as_ref()),
            EncodingRef::new_ref(RunEndEncoding.as_ref()),
            EncodingRef::new_ref(SequenceEncoding.as_ref()),
            EncodingRef::new_ref(SparseEncoding.as_ref()),
            EncodingRef::new_ref(ZigZagEncoding.as_ref()),
        ]);
        #[cfg(feature = "zstd")]
        arrays.register(vortex_zstd::ZstdEncoding.as_ref().into());

        // Register the layout encodings that Vortex ships with.
        let layouts = LayoutRegistry::default();

        // Register the expression encodings that Vortex ships with.
        let expressions = ExprRegistry::default();

        Self {
            arrays,
            layouts,
            expressions,
            metrics: VortexMetrics::default(),
        }
    }
}

impl VortexSession {
    /// Returns the array registry for this session.
    pub fn arrays(&self) -> &ArrayRegistry {
        &self.arrays
    }

    /// Returns the layout registry for this session.
    pub fn layouts(&self) -> &LayoutRegistry {
        &self.layouts
    }

    /// Returns the expression registry for this session.
    pub fn expressions(&self) -> &ExprRegistry {
        &self.expressions
    }

    /// Returns the metrics for this session.
    pub fn metrics(&self) -> &VortexMetrics {
        &self.metrics
    }
}
