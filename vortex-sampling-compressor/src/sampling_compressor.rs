use std::fmt::{Debug, Display, Formatter};

use log::{debug, warn};
use rand::rngs::StdRng;
use rand::SeedableRng;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::{ChunkedArray, Constant};
use vortex_array::compress::{
    check_dtype_unchanged, check_statistics_unchanged, check_validity_unchanged,
    CompressionStrategy,
};
use vortex_array::compute::slice;
use vortex_array::encoding::EncodingRef;
use vortex_array::validity::Validity;
use vortex_array::{ArrayDType, ArrayData, ArrayDef, IntoCanonical};
use vortex_error::{VortexExpect as _, VortexResult};

use crate::compressors::chunked::DEFAULT_CHUNKED_COMPRESSOR;
use crate::compressors::constant::ConstantCompressor;
use crate::compressors::struct_::StructCompressor;
use crate::compressors::{CompressedArray, CompressionTree, CompressorRef, EncodingCompressor};
use crate::sampling::stratified_slices;
use crate::{CompressConfig, Objective, DEFAULT_COMPRESSORS};

#[derive(Debug, Clone)]
pub struct SamplingCompressor<'a> {
    compressors: HashSet<CompressorRef<'a>>,
    options: CompressConfig,

    path: Vec<String>,
    depth: u8,
    /// A set of encodings disabled for this ctx.
    disabled_compressors: HashSet<CompressorRef<'a>>,
}

impl Display for SamplingCompressor<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}|{}]", self.depth, self.path.join("."))
    }
}

impl CompressionStrategy for SamplingCompressor<'_> {
    #[allow(clippy::same_name_method)]
    fn compress(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        Self::compress(self, array, None).map(CompressedArray::into_array)
    }

    fn used_encodings(&self) -> HashSet<EncodingRef> {
        self.compressors
            .iter()
            .flat_map(|c| c.used_encodings())
            .collect()
    }
}

impl Default for SamplingCompressor<'_> {
    fn default() -> Self {
        Self::new(HashSet::from(*DEFAULT_COMPRESSORS))
    }
}

impl<'a> SamplingCompressor<'a> {
    pub fn new(compressors: HashSet<CompressorRef<'a>>) -> Self {
        Self::new_with_options(compressors, Default::default())
    }

    pub fn new_with_options(
        compressors: HashSet<CompressorRef<'a>>,
        options: CompressConfig,
    ) -> Self {
        Self {
            compressors,
            options,
            path: Vec::new(),
            depth: 0,
            disabled_compressors: HashSet::new(),
        }
    }

    pub fn named(&self, name: &str) -> Self {
        let mut cloned = self.clone();
        cloned.path.push(name.into());
        cloned
    }

    // Returns a new ctx used for compressing an auxiliary array.
    // In practice, this means resetting any disabled encodings back to the original config.
    pub fn auxiliary(&self, name: &str) -> Self {
        let mut cloned = self.clone();
        cloned.path.push(name.into());
        cloned.disabled_compressors = HashSet::new();
        cloned
    }

    pub fn for_compressor(&self, compression: &dyn EncodingCompressor) -> Self {
        let mut cloned = self.clone();
        cloned.depth += compression.cost();
        cloned
    }

    #[inline]
    pub fn options(&self) -> &CompressConfig {
        &self.options
    }

    pub fn excluding(&self, compressor: CompressorRef<'a>) -> Self {
        let mut cloned = self.clone();
        cloned.disabled_compressors.insert(compressor);
        cloned
    }

    pub fn including(&self, compressor: CompressorRef<'a>) -> Self {
        let mut cloned = self.clone();
        cloned.compressors.insert(compressor);
        cloned
    }

    #[allow(clippy::same_name_method)]
    pub fn compress(
        &self,
        arr: &ArrayData,
        like: Option<&CompressionTree<'a>>,
    ) -> VortexResult<CompressedArray<'a>> {
        if arr.is_empty() {
            return Ok(CompressedArray::uncompressed(arr.clone()));
        }

        // Attempt to compress using the "like" array, otherwise fall back to sampled compression
        if let Some(l) = like {
            if let Some(compressed) = l.compress(arr, self) {
                let compressed = compressed?;

                check_validity_unchanged(arr, compressed.as_ref());
                check_dtype_unchanged(arr, compressed.as_ref());
                check_statistics_unchanged(arr, compressed.as_ref());
                return Ok(compressed);
            } else {
                warn!(
                    "{} cannot find compressor to compress {} like {}",
                    self, arr, l
                );
            }
        }

        // Otherwise, attempt to compress the array
        let compressed = self.compress_array(arr)?;

        check_validity_unchanged(arr, compressed.as_ref());
        check_dtype_unchanged(arr, compressed.as_ref());
        check_statistics_unchanged(arr, compressed.as_ref());
        Ok(compressed)
    }

    pub fn compress_validity(&self, validity: Validity) -> VortexResult<Validity> {
        match validity {
            Validity::Array(a) => Ok(Validity::Array(self.compress(&a, None)?.into_array())),
            a => Ok(a),
        }
    }

    pub fn compress_array(&self, array: &ArrayData) -> VortexResult<CompressedArray<'a>> {
        let mut rng = StdRng::seed_from_u64(self.options.rng_seed);

        if array.encoding().id() == Constant::ID {
            // Not much better we can do than constant!
            return Ok(CompressedArray::uncompressed(array.clone()));
        }

        if let Some(cc) = DEFAULT_CHUNKED_COMPRESSOR.can_compress(array) {
            return cc.compress(array, None, self.clone());
        }

        if let Some(cc) = StructCompressor.can_compress(array) {
            return cc.compress(array, None, self.clone());
        }

        if let Some(cc) = ConstantCompressor.can_compress(array) {
            return cc.compress(array, None, self.clone());
        }

        let (mut candidates, too_deep) = self
            .compressors
            .iter()
            .filter(|&encoding| !self.disabled_compressors.contains(encoding))
            .filter(|&encoding| encoding.can_compress(array).is_some())
            .partition::<Vec<&dyn EncodingCompressor>, _>(|&encoding| {
                self.depth + encoding.cost() <= self.options.max_cost
            });

        if !too_deep.is_empty() {
            debug!(
                "{} skipping encodings due to depth/cost: {}",
                self,
                too_deep
                    .iter()
                    .map(|x| x.id())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        debug!("{} candidates for {}: {:?}", self, array, candidates);

        if candidates.is_empty() {
            debug!(
                "{} no compressors for array with dtype: {} and encoding: {}",
                self,
                array.dtype(),
                array.encoding().id(),
            );
            return Ok(CompressedArray::uncompressed(array.clone()));
        }

        // We prefer all other candidates to the array's own encoding.
        // This is because we assume that the array's own encoding is the least efficient, but useful
        // to destructure an array in the final stages of compression. e.g. VarBin would be DictEncoded
        // but then the dictionary itself remains a VarBin array. DictEncoding excludes itself from the
        // dictionary, but we still have a large offsets array that should be compressed.
        // TODO(ngates): we actually probably want some way to prefer dict encoding over other varbin
        //  encodings, e.g. FSST.
        if candidates.len() > 1 {
            candidates.retain(|&compression| compression.id() != array.encoding().id().as_ref());
        }

        if array.len() <= (self.options.sample_size as usize * self.options.sample_count as usize) {
            // We're either already within a sample, or we're operating over a sufficiently small array.
            return find_best_compression(candidates, array, self);
        }

        // Take a sample of the array, then ask codecs for their best compression estimate.
        let sample = ChunkedArray::try_new(
            stratified_slices(
                array.len(),
                self.options.sample_size,
                self.options.sample_count,
                &mut rng,
            )
            .into_iter()
            .map(|(start, stop)| slice(array, start, stop))
            .collect::<VortexResult<Vec<ArrayData>>>()?,
            array.dtype().clone(),
        )?
        .into_canonical()?
        .into();

        let best = find_best_compression(candidates, &sample, self)?
            .into_path()
            .map(|best_compressor| {
                debug!(
                    "{} Compressing array {} with {}",
                    self, array, best_compressor
                );
                best_compressor.compress_unchecked(array, self)
            })
            .transpose()?;

        Ok(best.unwrap_or_else(|| CompressedArray::uncompressed(array.clone())))
    }
}

fn find_best_compression<'a>(
    candidates: Vec<&'a dyn EncodingCompressor>,
    sample: &ArrayData,
    ctx: &SamplingCompressor<'a>,
) -> VortexResult<CompressedArray<'a>> {
    let mut best = None;
    let mut best_objective = ctx.options().objective.starting_value();
    let mut best_objective_ratio = 1.0;
    // for logging
    let mut best_ratio = 1.0;
    let mut best_ratio_sample = None;

    for compression in candidates {
        debug!(
            "{} trying candidate {} for {}",
            ctx,
            compression.id(),
            sample
        );
        if compression.can_compress(sample).is_none() {
            continue;
        }
        let compressed_sample =
            compression.compress(sample, None, ctx.for_compressor(compression))?;

        let ratio = (compressed_sample.nbytes() as f64) / (sample.nbytes() as f64);
        let objective = Objective::evaluate(&compressed_sample, sample.nbytes(), ctx.options());

        // track the compression ratio, just for logging
        if ratio < best_ratio {
            best_ratio = ratio;

            // if we find one with a better compression ratio but worse objective value, save it
            // for debug logging later.
            if ratio < best_objective_ratio && objective >= best_objective {
                best_ratio_sample = Some(compressed_sample.clone());
            }
        }

        if objective < best_objective {
            best_objective = objective;
            best_objective_ratio = ratio;
            best = Some(compressed_sample);
        }

        debug!(
            "{} with {}: ratio ({}), objective fn value ({}); best so far: ratio ({}), objective fn value ({})",
            ctx,
            compression.id(),
            ratio,
            objective,
            best_ratio,
            best_objective
        );
    }

    let best = best.unwrap_or_else(|| CompressedArray::uncompressed(sample.clone()));
    if best_ratio < best_objective_ratio && best_ratio_sample.is_some() {
        let best_ratio_sample =
            best_ratio_sample.vortex_expect("already checked that this Option is Some");
        debug!(
            "{} best objective fn value ({}) has ratio {} from {}",
            ctx,
            best_objective,
            best_ratio,
            best.array().tree_display()
        );
        debug!(
            "{} best ratio ({}) has objective fn value {} from {}",
            ctx,
            best_ratio,
            best_objective,
            best_ratio_sample.array().tree_display()
        );
    }

    Ok(best)
}
