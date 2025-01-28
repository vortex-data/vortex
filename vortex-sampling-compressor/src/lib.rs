use std::sync::{Arc, LazyLock};

use compressors::bitpacked::BITPACK_WITH_PATCHES;
use compressors::chunked::DEFAULT_CHUNKED_COMPRESSOR;
use compressors::constant::ConstantCompressor;
use compressors::delta::DeltaCompressor;
use compressors::fsst::FSSTCompressor;
use compressors::struct_::StructCompressor;
use compressors::varbin::VarBinCompressor;
use compressors::{CompressedArray, CompressorRef};
use vortex_alp::{ALPEncoding, ALPRDEncoding};
use vortex_array::array::{
    ListEncoding, PrimitiveEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use vortex_array::encoding::EncodingRef;
use vortex_array::{Context, ContextRef};
use vortex_bytebool::ByteBoolEncoding;
use vortex_datetime_parts::DateTimePartsEncoding;
use vortex_dict::DictEncoding;
use vortex_fastlanes::{BitPackedEncoding, DeltaEncoding, FoREncoding};
use vortex_fsst::FSSTEncoding;
use vortex_runend::RunEndEncoding;
use vortex_zigzag::ZigZagEncoding;

use crate::compressors::alp::ALPCompressor;
use crate::compressors::date_time_parts::DateTimePartsCompressor;
use crate::compressors::dict::DictCompressor;
use crate::compressors::list::ListCompressor;
use crate::compressors::r#for::FoRCompressor;
use crate::compressors::runend::DEFAULT_RUN_END_COMPRESSOR;
use crate::compressors::sparse::SparseCompressor;
use crate::compressors::zigzag::ZigZagCompressor;

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod compressors;
mod constants;
mod downscale;
mod sampling;
mod sampling_compressor;

pub use sampling_compressor::*;
use vortex_sparse::SparseEncoding;

use crate::compressors::alp_rd::ALPRDCompressor;

pub const DEFAULT_COMPRESSORS: [CompressorRef; 15] = [
    &ALPCompressor as CompressorRef,
    &ALPRDCompressor,
    &BITPACK_WITH_PATCHES,
    &DEFAULT_CHUNKED_COMPRESSOR,
    &ConstantCompressor,
    &DateTimePartsCompressor,
    // &DeltaCompressor,
    &DictCompressor,
    &FoRCompressor,
    &FSSTCompressor,
    &DEFAULT_RUN_END_COMPRESSOR,
    &SparseCompressor,
    &StructCompressor,
    &ListCompressor,
    &VarBinCompressor,
    &ZigZagCompressor,
];

pub const ALL_COMPRESSORS: [CompressorRef; 16] = [
    &ALPCompressor as CompressorRef,
    &ALPRDCompressor,
    &BITPACK_WITH_PATCHES,
    &DEFAULT_CHUNKED_COMPRESSOR,
    &ConstantCompressor,
    &DateTimePartsCompressor,
    &DeltaCompressor,
    &DictCompressor,
    &FoRCompressor,
    &FSSTCompressor,
    &DEFAULT_RUN_END_COMPRESSOR,
    &SparseCompressor,
    &StructCompressor,
    &ListCompressor,
    &VarBinCompressor,
    &ZigZagCompressor,
];

pub static ALL_ENCODINGS_CONTEXT: LazyLock<ContextRef> = LazyLock::new(|| {
    Arc::new(Context::default().with_encodings([
        &ALPEncoding as EncodingRef,
        &ALPRDEncoding,
        &ByteBoolEncoding,
        &DateTimePartsEncoding,
        &DictEncoding,
        &BitPackedEncoding,
        &DeltaEncoding,
        &FoREncoding,
        &FSSTEncoding,
        &PrimitiveEncoding,
        &RunEndEncoding,
        &SparseEncoding,
        &StructEncoding,
        &ListEncoding,
        &VarBinEncoding,
        &VarBinViewEncoding,
        &ZigZagEncoding,
    ]))
});

#[derive(Debug, Clone)]
pub enum Objective {
    MinSize,
}

impl Objective {
    pub fn starting_value(&self) -> f64 {
        1.0
    }

    pub fn evaluate(
        array: &CompressedArray,
        base_size_bytes: usize,
        config: &CompressConfig,
    ) -> f64 {
        match &config.objective {
            Objective::MinSize => (array.nbytes() as f64) / (base_size_bytes as f64),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressConfig {
    /// Size of each sample slice
    sample_size: u16,
    /// Number of sample slices
    sample_count: u16,
    /// Random number generator seed
    rng_seed: u64,

    // Maximum depth of compression tree
    max_cost: u8,
    // Are we minimizing size or maximizing performance?
    objective: Objective,

    // Target chunk size in bytes
    target_block_bytesize: usize,
    // Target chunk size in row count
    target_block_size: usize,
}

impl CompressConfig {
    pub fn with_sample_size(mut self, sample_size: u16) -> Self {
        self.sample_size = sample_size;
        self
    }

    pub fn with_sample_count(mut self, sample_count: u16) -> Self {
        self.sample_count = sample_count;
        self
    }
}

impl Default for CompressConfig {
    fn default() -> Self {
        let kib = 1 << 10;
        let mib = 1 << 20;
        Self {
            // Sample length should always be multiple of 1024
            sample_size: 64,
            sample_count: 16,
            max_cost: constants::DEFAULT_MAX_COST,
            objective: Objective::MinSize,
            target_block_bytesize: 16 * mib,
            target_block_size: 64 * kib,
            rng_seed: 0,
        }
    }
}
