use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use itertools::{EitherOrBoth, Itertools};
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::encoding::EncodingRef;
use vortex_array::stats::{ArrayStatistics, Statistics};
use vortex_array::tree::TreeFormatter;
use vortex_array::ArrayData;
use vortex_error::{vortex_panic, VortexExpect, VortexResult};

use crate::SamplingCompressor;

pub mod alp;
pub mod alp_rd;
pub mod bitpacked;
pub mod chunked;
pub mod constant;
pub mod date_time_parts;
pub mod delta;
pub mod dict;
pub mod r#for;
pub mod fsst;
pub mod roaring_bool;
pub mod roaring_int;
pub mod runend;
pub mod runend_bool;
pub mod sparse;
pub mod struct_;
pub mod varbin;
pub mod zigzag;

pub trait EncodingCompressor: Sync + Send + Debug {
    fn id(&self) -> &str;

    fn cost(&self) -> u8;

    fn can_compress(&self, array: &ArrayData) -> Option<&dyn EncodingCompressor>;

    fn compress<'a>(
        &'a self,
        array: &ArrayData,
        like: Option<CompressionTree<'a>>,
        ctx: SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>>;

    fn used_encodings(&self) -> HashSet<EncodingRef>;
}

pub type CompressorRef<'a> = &'a dyn EncodingCompressor;

impl PartialEq for dyn EncodingCompressor + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}
impl Eq for dyn EncodingCompressor + '_ {}
impl Hash for dyn EncodingCompressor + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state)
    }
}

#[derive(Clone)]
pub struct CompressionTree<'a> {
    compressor: &'a dyn EncodingCompressor,
    children: Vec<Option<CompressionTree<'a>>>,
    metadata: Option<Arc<dyn EncoderMetadata>>,
}

impl Debug for CompressionTree<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// Metadata that can optionally be attached to a compression tree.
///
/// This enables codecs to cache trained parameters from the sampling runs to reuse for
/// the large run.
pub trait EncoderMetadata {
    fn as_any(&self) -> &dyn Any;
}

impl Display for CompressionTree<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut fmt = TreeFormatter::new(f, "".to_string());
        visit_child("root", Some(self), &mut fmt)
    }
}

fn visit_child(
    name: &str,
    child: Option<&CompressionTree>,
    fmt: &mut TreeFormatter,
) -> std::fmt::Result {
    fmt.indent(|f| {
        if let Some(child) = child {
            writeln!(f, "{name}: {}", child.compressor.id())?;
            for (i, c) in child.children.iter().enumerate() {
                visit_child(&format!("{name}.{}", i), c.as_ref(), f)?;
            }
        } else {
            writeln!(f, "{name}: uncompressed")?;
        }
        Ok(())
    })
}

impl<'a> CompressionTree<'a> {
    pub fn flat(compressor: &'a dyn EncodingCompressor) -> Self {
        Self::new(compressor, vec![])
    }

    pub fn new(
        compressor: &'a dyn EncodingCompressor,
        children: Vec<Option<CompressionTree<'a>>>,
    ) -> Self {
        Self {
            compressor,
            children,
            metadata: None,
        }
    }

    /// Save a piece of metadata as part of the compression tree.
    ///
    /// This can be specific encoder parameters that were discovered at sample time
    /// that should be reused when compressing the full array.
    pub(crate) fn new_with_metadata(
        compressor: &'a dyn EncodingCompressor,
        children: Vec<Option<CompressionTree<'a>>>,
        metadata: Arc<dyn EncoderMetadata>,
    ) -> Self {
        Self {
            compressor,
            children,
            metadata: Some(metadata),
        }
    }

    pub fn child(&self, idx: usize) -> Option<&CompressionTree<'a>> {
        self.children[idx].as_ref()
    }

    /// Compresses array with our compressor without verifying that the compressor can compress this array
    pub fn compress_unchecked(
        &self,
        array: &ArrayData,
        ctx: &SamplingCompressor<'a>,
    ) -> VortexResult<CompressedArray<'a>> {
        self.compressor.compress(
            array,
            Some(self.clone()),
            ctx.for_compressor(self.compressor),
        )
    }

    pub fn compress(
        &self,
        array: &ArrayData,
        ctx: &SamplingCompressor<'a>,
    ) -> Option<VortexResult<CompressedArray<'a>>> {
        self.compressor
            .can_compress(array)
            .map(|c| c.compress(array, Some(self.clone()), ctx.for_compressor(c)))
    }

    pub fn compressor(&self) -> &dyn EncodingCompressor {
        self.compressor
    }

    /// Access the saved opaque metadata.
    ///
    /// This will consume the owned metadata, giving the caller ownership of
    /// the Box.
    ///
    /// The value of `T` will almost always be `EncodingCompressor`-specific.
    pub fn metadata(&mut self) -> Option<Arc<dyn EncoderMetadata>> {
        std::mem::take(&mut self.metadata)
    }

    pub fn num_descendants(&self) -> usize {
        self.children
            .iter()
            .filter_map(|child| child.as_ref().map(|c| c.num_descendants() + 1))
            .sum::<usize>()
    }

    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        &'a dyn EncodingCompressor,
        Vec<Option<CompressionTree<'a>>>,
        Option<Arc<dyn EncoderMetadata>>,
    ) {
        (self.compressor, self.children, self.metadata)
    }
}

#[derive(Debug, Clone)]
pub struct CompressedArray<'a> {
    array: ArrayData,
    path: Option<CompressionTree<'a>>,
}

impl<'a> CompressedArray<'a> {
    pub fn uncompressed(array: ArrayData) -> Self {
        Self { array, path: None }
    }

    pub fn compressed(
        array: ArrayData,
        path: Option<CompressionTree<'a>>,
        stats_to_inherit: Option<&dyn Statistics>,
    ) -> Self {
        if let Some(stats) = stats_to_inherit {
            // eagerly compute uncompressed size in bytes at compression time, since it's
            // too expensive to compute after compression
            let _ = stats.compute_uncompressed_size_in_bytes();
            array.inherit_statistics(stats);
        }
        let compressed = Self { array, path };
        compressed.validate();
        compressed
    }

    fn validate(&self) {
        self.validate_children(self.path.as_ref(), &self.array)
    }

    fn validate_children(&self, path: Option<&CompressionTree>, array: &ArrayData) {
        if let Some(path) = path.as_ref() {
            path.children
                .iter()
                .zip_longest(array.children().iter())
                .for_each(|pair| match pair {
                    EitherOrBoth::Both(Some(child_tree), child_array) => {
                        self.validate_children(Some(child_tree), child_array);
                    }
                    EitherOrBoth::Left(Some(child_tree)) => {
                        vortex_panic!(
                            "Child tree without child array!!\nroot tree: {}\nroot array: {}\nlocal tree: {path}\nlocal array: {}\nproblematic child_tree: {child_tree}",
                            self.path().as_ref().vortex_expect("must be present"),
                            self.array.tree_display(),
                            array.tree_display()
                        );
                    }
                    // if the child_tree is None, we have an uncompressed child array or both were None; fine either way
                    _ => {},
                });
        }
    }

    #[inline]
    pub fn array(&self) -> &ArrayData {
        &self.array
    }

    #[inline]
    pub fn into_array(self) -> ArrayData {
        self.array
    }

    #[inline]
    pub fn path(&self) -> &Option<CompressionTree> {
        &self.path
    }

    #[inline]
    pub fn into_path(self) -> Option<CompressionTree<'a>> {
        self.path
    }

    #[inline]
    pub fn into_parts(self) -> (ArrayData, Option<CompressionTree<'a>>) {
        (self.array, self.path)
    }

    /// Total size of the array in bytes, including all children and buffers.
    #[inline]
    pub fn nbytes(&self) -> usize {
        self.array.nbytes()
    }
}

impl AsRef<ArrayData> for CompressedArray<'_> {
    fn as_ref(&self) -> &ArrayData {
        &self.array
    }
}
