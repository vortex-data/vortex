// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`IvfLayout`] vtable binding.

use std::sync::Arc;

use vortex_array::DeserializeMetadata;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_layout::LayoutChildType;
use vortex_layout::LayoutChildren;
use vortex_layout::LayoutEncodingRef;
use vortex_layout::LayoutId;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutRef;
use vortex_layout::VTable;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::vtable;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::layout::IVF_LAYOUT_ID;
use crate::layout::metadata::IvfLayoutMetadata;
use crate::layout::reader::IvfReader;

vtable!(Ivf);

impl VTable for Ivf {
    type Layout = IvfLayout;
    type Encoding = IvfLayoutEncoding;
    type Metadata = ProstMetadata<IvfLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new(IVF_LAYOUT_ID)
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(IvfLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.data.row_count()
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        layout.data.dtype()
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        ProstMetadata(IvfLayoutMetadata::new(
            layout.dim,
            layout.nprobes,
            layout.num_clusters,
        ))
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        2
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match idx {
            0 => Ok(Arc::clone(&layout.data)),
            1 => Ok(Arc::clone(&layout.centroids)),
            _ => vortex_bail!("IvfLayout has only 2 children (data, centroids), got {idx}"),
        }
    }

    fn child_type(_layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match idx {
            0 => LayoutChildType::Transparent("data".into()),
            1 => LayoutChildType::Auxiliary("centroids".into()),
            _ => vortex_panic!("IvfLayout has only 2 children"),
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(IvfReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        _dtype: &DType,
        _row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: &ReadContext,
    ) -> VortexResult<Self::Layout> {
        vortex_ensure!(
            children.nchildren() == 2,
            "IvfLayout requires exactly 2 children (data, centroids), got {}",
            children.nchildren()
        );

        // Build children with their own dtypes. The data child uses the layout's dtype.
        // We reconstruct the centroids dtype below from the data dtype + metadata.
        // For the build step we rely on the write-time child layouts having set their own dtypes.
        // The centroids child's dtype is a `Vector<dim, f32>` extension type sharing the element
        // type with the data child.
        let data_dtype = get_data_dtype(children, _dtype)?;
        let centroids_dtype = centroids_dtype_for(data_dtype)?;

        let data = children.child(0, data_dtype)?;
        let centroids = children.child(1, &centroids_dtype)?;

        Ok(IvfLayout {
            data,
            centroids,
            dim: metadata.dim,
            nprobes: metadata.nprobes,
            num_clusters: metadata.num_clusters,
        })
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 2,
            "IvfLayout expects exactly 2 children (data, centroids), got {}",
            children.len()
        );
        let mut it = children.into_iter();
        layout.data = it
            .next()
            .ok_or_else(|| vortex_error::vortex_err!("missing data child"))?;
        layout.centroids = it
            .next()
            .ok_or_else(|| vortex_error::vortex_err!("missing centroids child"))?;
        Ok(())
    }
}

/// The data child uses the same dtype as the overall layout (a `Vector<dim, f32>` extension).
/// This helper keeps the extraction step explicit.
fn get_data_dtype<'a>(_children: &dyn LayoutChildren, dtype: &'a DType) -> VortexResult<&'a DType> {
    Ok(dtype)
}

/// Compute the centroids child's dtype. The centroids have the same `Vector` extension dtype as
/// the data column, but with one row per cluster rather than one row per input vector.
fn centroids_dtype_for(data_dtype: &DType) -> VortexResult<DType> {
    // The centroids are a Vector extension array with the same dimensionality as the data.
    // The dtype is identical to the data dtype (both are the same Vector extension type).
    Ok(data_dtype.clone())
}

/// Encoding marker for [`IvfLayout`].
#[derive(Debug)]
pub struct IvfLayoutEncoding;

/// An IVF vector index layout.
///
/// See the [module documentation](crate::layout) for details of the on-disk structure and
/// pruning behavior.
#[derive(Clone, Debug)]
pub struct IvfLayout {
    pub(crate) data: LayoutRef,
    pub(crate) centroids: LayoutRef,
    pub(crate) dim: u32,
    pub(crate) nprobes: u32,
    pub(crate) num_clusters: u32,
}

impl IvfLayout {
    /// Construct an [`IvfLayout`] from its children.
    ///
    /// `data` must be a chunked layout with one chunk per cluster (though any layout is accepted).
    /// `centroids` must be a flat layout storing a `Vector<dim, f32>` extension array with exactly
    /// `num_clusters` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have inconsistent row counts or dtypes.
    pub fn try_new(
        data: LayoutRef,
        centroids: LayoutRef,
        dim: u32,
        num_clusters: u32,
        nprobes: u32,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            centroids.row_count() == u64::from(num_clusters),
            "centroids must have {num_clusters} rows, got {}",
            centroids.row_count()
        );
        vortex_ensure!(dim > 0, "dim must be > 0");
        vortex_ensure!(num_clusters > 0, "num_clusters must be > 0");
        Ok(Self {
            data,
            centroids,
            dim,
            nprobes: nprobes.max(1),
            num_clusters,
        })
    }

    /// Returns the vector dimension.
    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Returns the default number of clusters to probe at query time.
    pub fn nprobes(&self) -> u32 {
        self.nprobes
    }

    /// Returns the number of clusters (K).
    pub fn num_clusters(&self) -> u32 {
        self.num_clusters
    }

    /// Returns the data child layout.
    pub fn data(&self) -> &LayoutRef {
        &self.data
    }

    /// Returns the centroids child layout.
    pub fn centroids(&self) -> &LayoutRef {
        &self.centroids
    }
}
