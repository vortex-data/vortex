use std::sync::{Arc, LazyLock};

use compressors::bitpacked::BITPACK_WITH_PATCHES;
use compressors::chunked::DEFAULT_CHUNKED_COMPRESSOR;
use compressors::constant::ConstantCompressor;
use compressors::delta::DeltaCompressor;
use compressors::fsst::FSSTCompressor;
use compressors::roaring_bool::RoaringBoolCompressor;
use compressors::roaring_int::RoaringIntCompressor;
use compressors::struct_::StructCompressor;
use compressors::varbin::VarBinCompressor;
use compressors::{CompressedArray, CompressionTree, CompressorRef};
use vortex_alp::{ALPEncoding, ALPRDEncoding};
use vortex_array::array::{
    PrimitiveEncoding, SparseEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use vortex_array::encoding::EncodingRef;
use vortex_array::Context;
use vortex_bytebool::ByteBoolEncoding;
use vortex_datetime_parts::DateTimePartsEncoding;
use vortex_dict::DictEncoding;
use vortex_fastlanes::{BitPackedEncoding, DeltaEncoding, FoREncoding};
use vortex_fsst::FSSTEncoding;
use vortex_roaring::{RoaringBoolEncoding, RoaringIntEncoding};
use vortex_runend::RunEndEncoding;
use vortex_runend_bool::RunEndBoolEncoding;
use vortex_zigzag::ZigZagEncoding;

use crate::compressors::alp::ALPCompressor;
use crate::compressors::date_time_parts::DateTimePartsCompressor;
use crate::compressors::dict::DictCompressor;
use crate::compressors::r#for::FoRCompressor;
use crate::compressors::runend::DEFAULT_RUN_END_COMPRESSOR;
use crate::compressors::runend_bool::RunEndBoolCompressor;
use crate::compressors::sparse::SparseCompressor;
use crate::compressors::zigzag::ZigZagCompressor;

#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod compressors;
mod constants;
mod sampling;
mod sampling_compressor;

pub use sampling_compressor::*;

pub const DEFAULT_COMPRESSORS: [CompressorRef; 14] = [
    &ALPCompressor as CompressorRef,
    &BITPACK_WITH_PATCHES,
    &DEFAULT_CHUNKED_COMPRESSOR,
    &ConstantCompressor,
    &DateTimePartsCompressor,
    // &DeltaCompressor,
    &DictCompressor,
    &FoRCompressor,
    &FSSTCompressor,
    //&RoaringBoolCompressor,
    //&RoaringIntCompressor,
    &RunEndBoolCompressor,
    &DEFAULT_RUN_END_COMPRESSOR,
    &SparseCompressor,
    &StructCompressor,
    &VarBinCompressor,
    &ZigZagCompressor,
];

pub const ALL_COMPRESSORS: [CompressorRef; 17] = [
    &ALPCompressor as CompressorRef,
    &BITPACK_WITH_PATCHES,
    &DEFAULT_CHUNKED_COMPRESSOR,
    &ConstantCompressor,
    &DateTimePartsCompressor,
    &DeltaCompressor,
    &DictCompressor,
    &FoRCompressor,
    &FSSTCompressor,
    &RoaringBoolCompressor,
    &RoaringIntCompressor,
    &RunEndBoolCompressor,
    &DEFAULT_RUN_END_COMPRESSOR,
    &SparseCompressor,
    &StructCompressor,
    &VarBinCompressor,
    &ZigZagCompressor,
];

pub static ALL_ENCODINGS_CONTEXT: LazyLock<Arc<Context>> = LazyLock::new(|| {
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
        &RoaringBoolEncoding,
        &RoaringIntEncoding,
        &RunEndEncoding,
        &RunEndBoolEncoding,
        &SparseEncoding,
        &StructEncoding,
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
        let num_descendants = array
            .path()
            .as_ref()
            .map(CompressionTree::num_descendants)
            .unwrap_or(0) as u64;
        let overhead_bytes = num_descendants * config.overhead_bytes_per_array;
        let size_in_bytes = array.nbytes() as u64 + overhead_bytes;

        match &config.objective {
            Objective::MinSize => (size_in_bytes as f64) / (base_size_bytes as f64),
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
    /// Penalty in bytes per compression level
    overhead_bytes_per_array: u64,

    // Target chunk size in bytes
    target_block_bytesize: usize,
    // Target chunk size in row count
    target_block_size: usize,
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
            overhead_bytes_per_array: 64,
            target_block_bytesize: 16 * mib,
            target_block_size: 64 * kib,
            rng_seed: 0,
        }
    }
}
