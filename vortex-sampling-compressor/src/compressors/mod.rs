use std::any::Any;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use vortex_array::aliases::hash_set::HashSet;
use vortex_array::encoding::EncodingRef;
use vortex_array::stats::ArrayStatistics;
use vortex_array::tree::TreeFormatter;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::{ArrayDType, ArrayData, NamedChildrenCollector};
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
pub mod list;
#[cfg(not(target_arch = "wasm32"))]
pub mod roaring_bool;
#[cfg(not(target_arch = "wasm32"))]
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
    name: Option<&'a str>,
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
            writeln!(f, "{name}|{:?}: {}", child.name, child.compressor.id())?;
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

    pub fn named(&mut self, name: &'a str) {
        self.name = Some(name);
    }

    pub fn new(
        compressor: &'a dyn EncodingCompressor,
        children: Vec<Option<CompressionTree<'a>>>,
    ) -> Self {
        println!("new {}, {:?}", compressor.id(), children);
        Self {
            compressor,
            name: None,
            children,
            metadata: None,
        }
    }

    pub fn new_named(
        compressor: &'a dyn EncodingCompressor,
        name: &'a str,
        children: Vec<Option<CompressionTree<'a>>>,
    ) -> Self {
        println!("new_named {:?}, {:?}", name, children);
        Self {
            compressor,
            name: Some(name),
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
        name: Option<&'a str>,
        children: Vec<Option<CompressionTree<'a>>>,
        metadata: Arc<dyn EncoderMetadata>,
    ) -> Self {
        println!("new_with_metadata {:?}, {:?}", name, children);
        Self {
            compressor,
            name,
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
    pub array: ArrayData,
    pub name: Option<Arc<str>>,
    pub path: Option<CompressionTree<'a>>,
}

impl<'a> CompressedArray<'a> {
    pub fn uncompressed(array: ArrayData) -> Self {
        Self {
            array,
            name: None,
            path: None,
        }
    }

    pub fn compressed(
        compressed: ArrayData,
        path: Option<CompressionTree<'a>>,
        uncompressed: impl AsRef<ArrayData>,
    ) -> Self {
        let uncompressed = uncompressed.as_ref();

        // Sanity check the compressed array
        assert_eq!(
            compressed.len(),
            uncompressed.len(),
            "Compressed array {} has different length to uncompressed",
            compressed.encoding().id(),
        );
        assert_eq!(
            compressed.dtype(),
            uncompressed.dtype(),
            "Compressed array {} has different dtype to uncompressed",
            compressed.encoding().id(),
        );

        // eagerly compute uncompressed size in bytes at compression time, since it's
        // too expensive to compute after compression
        let _ = uncompressed
            .statistics()
            .compute_uncompressed_size_in_bytes();
        compressed.inherit_statistics(uncompressed.statistics());

        let compressed = Self {
            array: compressed,
            name: None,
            path,
        };
        compressed.validate();
        compressed
    }

    fn validate(&self) {
        // self.validate_children(self.path.as_ref(), &self.array)
    }

    #[allow(dead_code)]
    fn validate_children(&self, path: Option<&CompressionTree>, array: &ArrayData) {
        println!("val {}", self.array.tree_display());
        println!("path {:?}", path);
        println!("arr {:?}", array.encoding());
        let mut col = Box::new(NamedChildrenCollector::new_with_depth(1));
        array
            .encoding()
            .accept(array, col.as_mut() as &mut dyn ArrayVisitor)
            .vortex_expect("Failed to get children");

        let map2: HashMap<_, _> = col
            .children()
            .into_iter()
            .map(|v| (v.0.as_str(), &v.1))
            .collect();
        println!("arr map {:?}", map2.keys());

        if let Some(path) = path.as_ref() {
            println!("path ch {:?}", path.children);
            path.children
                .iter()
                .filter_map(|path| path.as_ref())
                .map(|tree| {
                    println!("tree: {:?}", tree);
                    println!("tree: {:?}", tree.name);
                    println!("arr: {}", array.tree_display());
                    // let name = tree.name.expect("all children must be named");
                    let name = tree.name.unwrap_or("");
                    let v2 = map2.get(&name);
                    (tree, v2)
                })
                .filter(|(_a, b)| b.is_some())
                .for_each(|(tree, arr)| {
                    let Some(arr) = arr else {
                        vortex_panic!(
                            "compress tree node {:?} doesn't exist in array {:?}",
                            tree.name,
                            map2.keys()
                        )
                    };
                    self.validate_children(Some(tree), arr);
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

    pub fn named(&mut self, name: Arc<str>) {
        self.name = Some(name);
        println!("set name {:?}", self.name);
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
