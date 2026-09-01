//! A prototype of the `vortex.indexed` layout proposed in
//! [vortex-data/vortex#9024](https://github.com/vortex-data/vortex/issues/9024).
//!
//! A `vortex.indexed` layout wraps a data layout with zero or more *locating indexes* held as
//! auxiliary children. Writers build indexes through the pluggable [`IndexVTable`] registry while
//! streaming; readers probe those indexes to prune (or outright answer) filter predicates, and
//! fall back to a plain scan of the data child whenever an index is missing, unknown, or has no
//! claim on the expression.
//!
//! Indexes are optional at every stage. A builder that finds nothing worth keeping declines once
//! the stream is drained — see [`IndexBuilder::finish`] — and if every builder declines, no wrapper
//! is written at all and the file carries the plain data layout.
//!
//! # Shape
//!
//! ```text
//! vortex.indexed
//! ├── child 0: data      Transparent("data")
//! ├── child 1: index #0  Auxiliary("index:<id>")
//! └── child n: index #n  Auxiliary("index:<id>")
//! ```
//!
//! Index content is written through an ordinary layout strategy, so it is chunked, zone-mapped,
//! and compressed by the same machinery as data. Probing an index is therefore just a pruned scan
//! over the index child: a sorted key column's zone map narrows the probe to a handful of zones.
//!
//! # What ships here
//!
//! The generic wrapper only: [`Indexed`], [`writer::IndexedStrategy`] and [`reader::IndexedReader`],
//! plus the [`IndexVTable`] contract that index kinds implement. Concrete kinds are registered into
//! an [`session::IndexSession`] — see the `reverse_index` example (`examples/reverse_index/`) for
//! a worked example, a minimal equality index over any column.

pub mod index;
pub(crate) mod reader;
pub mod session;
pub mod writer;

use std::sync::Arc;

use prost::Message;
use vortex_array::DeserializeMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

pub use self::index::IndexBuilder;
pub use self::index::IndexExactness;
pub use self::index::IndexId;
pub use self::index::IndexQueryPlan;
pub use self::index::IndexResolve;
pub use self::index::IndexVTable;
pub use self::index::IndexVTableRef;
pub use self::index::RowLocator;
pub use self::session::IndexSession;
pub use self::session::IndexSessionExt;
pub use self::writer::IndexConfig;
pub use self::writer::IndexedStrategy;
use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::layout_children;
use crate::layouts::indexed::reader::IndexedReader;
use crate::segments::SegmentSource;

/// Registry id of this layout encoding.
pub const INDEXED_LAYOUT_ID: &str = "vortex.indexed";

/// Leading byte of the serialized metadata, so the protobuf can be re-shaped later.
const INDEXED_METADATA_VERSION: u8 = 1;

/// Layout vtable for the `vortex.indexed` layout.
#[derive(Clone, Debug)]
pub struct Indexed;

/// One index attached to the data child.
#[derive(Clone, Debug)]
pub struct IndexSpec {
    id: IndexId,
    options: Arc<[u8]>,
    index_dtype: DType,
    /// The resolved kind, or `None` when it is not registered in this session. An unresolved spec
    /// is inert: its child is never probed and reads fall through to the data child.
    vtable: Option<IndexVTableRef>,
}

impl IndexSpec {
    /// Create a spec for an index that was just built by `vtable`.
    pub fn new(vtable: IndexVTableRef, options: Vec<u8>, index_dtype: DType) -> Self {
        Self {
            id: vtable.id(),
            options: options.into(),
            index_dtype,
            vtable: Some(vtable),
        }
    }

    /// The registry id of this index's kind.
    pub fn id(&self) -> IndexId {
        self.id
    }

    /// The kind-defined, self-versioned options blob.
    pub fn options(&self) -> &[u8] {
        &self.options
    }

    /// The dtype of this index's layout child.
    pub fn index_dtype(&self) -> &DType {
        &self.index_dtype
    }

    /// The resolved kind, or `None` if it is not registered in this session.
    pub fn vtable(&self) -> Option<&IndexVTableRef> {
        self.vtable.as_ref()
    }
}

/// Layout-specific data for the [`IndexedLayout`].
///
/// Child 0 is the data, sharing this layout's dtype and row space. Children 1.. are index content,
/// one per entry in [`IndexedData::indexes`].
#[derive(Clone, Debug)]
pub struct IndexedData {
    indexes: Arc<[IndexSpec]>,
}

/// A layout that attaches locating indexes to a data child.
pub type IndexedLayout = Layout<Indexed>;

impl IndexedLayout {
    /// Assemble an indexed layout from a data child and one layout child per index spec.
    pub fn try_new(
        data: LayoutRef,
        index_layouts: Vec<LayoutRef>,
        indexes: Vec<IndexSpec>,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            index_layouts.len() == indexes.len(),
            "IndexedLayout got {} index children for {} specs",
            index_layouts.len(),
            indexes.len()
        );
        for (layout, spec) in index_layouts.iter().zip(&indexes) {
            vortex_ensure!(
                layout.dtype() == &spec.index_dtype,
                "Index child dtype {} does not match spec dtype {} for {}",
                layout.dtype(),
                spec.index_dtype,
                spec.id
            );
        }

        let dtype = data.dtype().clone();
        let row_count = data.row_count();
        let mut children = Vec::with_capacity(1 + index_layouts.len());
        children.push(data);
        children.extend(index_layouts);

        Ok(LayoutParts::new(
            Indexed,
            dtype,
            row_count,
            Vec::new(),
            layout_children(children),
            IndexedData {
                indexes: indexes.into(),
            },
        )
        .into_typed())
    }

    /// The indexes attached to the data child.
    pub fn indexes(&self) -> &Arc<[IndexSpec]> {
        &self.indexes
    }
}

impl VTable for Indexed {
    type LayoutData = IndexedData;
    type Metadata = IndexedMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new(INDEXED_LAYOUT_ID);
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        IndexedMetadata {
            indexes: layout
                .indexes
                .iter()
                .map(|spec| IndexSpecProto {
                    id: spec.id.to_string(),
                    options: spec.options.to_vec(),
                    index_dtype: Some(
                        pb::DType::try_from(&spec.index_dtype)
                            .vortex_expect("index child dtype should be serializable"),
                    ),
                })
                .collect::<Vec<_>>()
                .into(),
        }
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &IndexedMetadata,
    ) -> VortexResult<Self::LayoutData> {
        vortex_ensure!(
            args.children.nchildren() == 1 + metadata.indexes.len(),
            "IndexedLayout expects {} children (data + {} indexes), got {}",
            1 + metadata.indexes.len(),
            metadata.indexes.len(),
            args.children.nchildren()
        );

        let registry = args.session.indexes();
        let indexes = metadata
            .indexes
            .iter()
            .map(|spec| {
                let index_dtype = spec
                    .index_dtype
                    .as_ref()
                    .map(|dtype| DType::from_proto(dtype, args.session))
                    .transpose()?
                    .ok_or_else(|| vortex_err!("Index spec {} is missing its dtype", spec.id))?;
                let id = IndexId::from(spec.id.as_str());

                // An unknown kind degrades to an inert spec rather than failing the read: the
                // child stays addressable so child counts and dtypes still line up, but nothing
                // ever probes it.
                Ok(IndexSpec {
                    id,
                    options: spec.options.as_slice().into(),
                    index_dtype,
                    vtable: registry.find(&id),
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;

        args.children.child(0, args.dtype)?;
        for (idx, spec) in indexes.iter().enumerate() {
            args.children.child(idx + 1, &spec.index_dtype)?;
        }

        Ok(IndexedData {
            indexes: indexes.into(),
        })
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        match slot {
            0 => Ok(layout.dtype().clone()),
            _ => {
                let Some(spec) = layout.indexes.get(slot - 1) else {
                    vortex_bail!("Invalid child index: {}", slot);
                };
                Ok(spec.index_dtype.clone())
            }
        }
    }

    fn child_type(layout: &Layout<Self>, slot: usize) -> LayoutChildType {
        match slot {
            0 => LayoutChildType::Transparent("data".into()),
            _ => match layout.indexes.get(slot - 1) {
                Some(spec) => LayoutChildType::Auxiliary(format!("index:{}", spec.id).into()),
                None => vortex_panic!("Invalid child index: {}", slot),
            },
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(IndexedReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx.clone(),
        )?))
    }
}

/// Serialized indexed-layout metadata: one entry per index child, in child order.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexedMetadata {
    indexes: Arc<[IndexSpecProto]>,
}

#[derive(Clone, PartialEq, Message)]
struct IndexedMetadataProto {
    #[prost(message, repeated, tag = "1")]
    indexes: Vec<IndexSpecProto>,
}

#[derive(Clone, PartialEq, Message)]
struct IndexSpecProto {
    /// Registry id of the index kind, e.g. `vortex.idx.reverse_index`.
    #[prost(string, tag = "1")]
    id: String,
    /// Kind-defined, self-versioned options.
    #[prost(bytes = "vec", tag = "2")]
    options: Vec<u8>,
    /// The dtype of the index child. Layout nodes carry no dtype of their own — it flows top-down
    /// during deserialization — so an auxiliary child's dtype has to be recorded here.
    #[prost(message, optional, tag = "3")]
    index_dtype: Option<pb::DType>,
}

impl SerializeMetadata for IndexedMetadata {
    fn serialize(self) -> Vec<u8> {
        let proto = IndexedMetadataProto {
            indexes: self.indexes.to_vec(),
        };
        let mut metadata = vec![INDEXED_METADATA_VERSION];
        metadata.extend(proto.encode_to_vec());
        metadata
    }
}

impl DeserializeMetadata for IndexedMetadata {
    type Output = Self;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        let Some((&version, proto_bytes)) = metadata.split_first() else {
            vortex_bail!("Indexed metadata missing protobuf version");
        };
        vortex_ensure!(
            version == INDEXED_METADATA_VERSION,
            "Unsupported indexed metadata version: {}",
            version
        );

        let proto = IndexedMetadataProto::decode(proto_bytes)
            .map_err(|err| vortex_err!("Failed to decode indexed metadata: {err}"))?;
        Ok(Self {
            indexes: proto.indexes.into(),
        })
    }
}

#[cfg(test)]
mod tests;
