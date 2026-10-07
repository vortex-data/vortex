// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A layout that stores Parquet Variant columns as their decomposed storage struct.
//!
//! A [`ParquetVariantLayout`] has a single child holding the Parquet Variant storage
//! `{metadata, value, typed_value}` as a struct, so the table writer splits it into one column per
//! storage child: every shredded `typed_value` leaf becomes its own column with its own zone
//! statistics, dictionary, and compression.
//!
//! The reader rewrites `variant_get` expressions over the Variant column into expressions over
//! the storage struct (see [`reader`]). A path whose shredded leaf fully represents its values is
//! served from that leaf's typed column alone, so filters and projections on shredded paths read
//! and evaluate only the columns they need.

mod expr;
mod reader;
#[cfg(test)]
mod tests;
mod writer;

use std::sync::Arc;

use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::proto::dtype as pb;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_layout::Layout;
use vortex_layout::LayoutChildType;
use vortex_layout::LayoutDeserializeArgs;
use vortex_layout::LayoutEncodingRef;
use vortex_layout::LayoutId;
use vortex_layout::LayoutParts;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutRef;
use vortex_layout::VTable;
use vortex_layout::layout_children;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
pub use writer::ParquetVariantLayoutStrategy;

pub(crate) use self::expr::ParquetVariantFromStorage;
use crate::editions;

/// Registers the layout and its edition with `session`, and makes table writers write Variant
/// columns as [`ParquetVariantLayout`]s.
pub(crate) fn initialize(session: &VortexSession) {
    session
        .layouts()
        .register(LayoutEncodingRef::new_ref(&ParquetVariantLayoutEncoding));
    session.scalar_fns().register(ParquetVariantFromStorage);
    // `initialize` is idempotent, so a repeated call must not re-declare the edition.
    if session
        .editions()
        .find(&editions::VARIANT_2026_08)
        .is_none()
    {
        if session
            .editions()
            .find_family(editions::FAMILY.name)
            .is_none()
        {
            session
                .editions()
                .declare_family(&editions::FAMILY)
                .map_err(|error| vortex_err!("{error}"))
                .vortex_expect("variant edition family is valid");
        }
        session
            .register_edition(&editions::DECLARATION)
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("variant edition declaration is valid");
    }
    session
        .enable_edition(editions::VARIANT_2026_08)
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("variant edition is registered");
    session
        .layouts()
        .register_variant_strategy(Arc::new(|storage| {
            Arc::new(ParquetVariantLayoutStrategy::new(storage)) as _
        }));
}

/// Parquet Variant layout vtable.
#[derive(Clone, Debug)]
pub struct ParquetVariantLayoutEncoding;

/// A Variant column stored as its decomposed Parquet Variant storage struct.
pub type ParquetVariantLayout = Layout<ParquetVariantLayoutEncoding>;

/// A shredded object path, as the field names leading to it.
pub(crate) type ShreddedPath = Vec<FieldName>;

/// Parquet-Variant-layout-specific data.
#[derive(Clone, Debug)]
pub struct ParquetVariantLayoutData {
    storage_dtype: DType,
    /// Shredded object paths whose residual `value` is null in every row, so their `typed_value`
    /// alone represents every value at the path.
    typed_paths: Arc<[ShreddedPath]>,
}

impl ParquetVariantLayoutData {
    /// The dtype of the storage struct child.
    pub fn storage_dtype(&self) -> &DType {
        &self.storage_dtype
    }

    pub(crate) fn typed_paths(&self) -> &[ShreddedPath] {
        &self.typed_paths
    }
}

/// Serialized metadata of a [`ParquetVariantLayout`].
#[derive(prost::Message)]
pub struct ParquetVariantLayoutMetadata {
    /// The dtype of the storage struct child.
    #[prost(message, optional, tag = "1")]
    pub storage_dtype: Option<pb::DType>,
    /// Shredded object paths whose residual `value` is null in every row.
    #[prost(message, repeated, tag = "2")]
    pub typed_paths: Vec<TypedPathMetadata>,
}

/// A shredded object path in [`ParquetVariantLayoutMetadata`].
#[derive(Clone, prost::Message)]
pub struct TypedPathMetadata {
    /// The object field names leading to the path.
    #[prost(string, repeated, tag = "1")]
    pub fields: Vec<String>,
}

impl VTable for ParquetVariantLayoutEncoding {
    type LayoutData = ParquetVariantLayoutData;
    type Metadata = ProstMetadata<ParquetVariantLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.parquet_variant");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        ProstMetadata(ParquetVariantLayoutMetadata {
            storage_dtype: Some(
                (&layout.storage_dtype)
                    .try_into()
                    .unwrap_or_else(|e| vortex_panic!("cannot serialize storage dtype: {e}")),
            ),
            typed_paths: layout
                .typed_paths
                .iter()
                .map(|path| TypedPathMetadata {
                    fields: path.iter().map(|name| name.as_ref().to_string()).collect(),
                })
                .collect(),
        })
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &ParquetVariantLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        vortex_ensure!(
            args.dtype.is_variant(),
            "ParquetVariantLayout dtype must be Variant, found {}",
            args.dtype
        );
        vortex_ensure_eq!(
            args.children.nchildren(),
            1,
            "ParquetVariantLayout expects exactly one storage child"
        );
        let storage_dtype = metadata
            .storage_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("ParquetVariantLayout metadata is missing storage dtype"))
            .and_then(|dtype| DType::from_proto(dtype, args.session))?;
        validate_storage_dtype(args.dtype, &storage_dtype)?;
        vortex_ensure_eq!(
            args.children.child_row_count(0),
            args.row_count,
            "ParquetVariantLayout storage row count does not match parent"
        );
        Ok(ParquetVariantLayoutData {
            storage_dtype,
            typed_paths: metadata
                .typed_paths
                .iter()
                .map(|path| {
                    path.fields
                        .iter()
                        .map(|f| FieldName::from(f.as_str()))
                        .collect()
                })
                .collect(),
        })
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        match slot {
            0 => Ok(layout.storage_dtype.clone()),
            _ => vortex_bail!("ParquetVariantLayout has no child {slot}"),
        }
    }

    fn child_type(_layout: &Layout<Self>, slot: usize) -> LayoutChildType {
        match slot {
            0 => LayoutChildType::Transparent("storage".into()),
            _ => vortex_panic!("ParquetVariantLayout has no child {slot}"),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(reader::ParquetVariantReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx.clone(),
        )?))
    }
}

/// Creates a Parquet Variant layout over its storage struct child.
pub(crate) fn new_parquet_variant_layout(
    dtype: DType,
    storage: LayoutRef,
    typed_paths: Vec<ShreddedPath>,
) -> VortexResult<ParquetVariantLayout> {
    {
        let storage_dtype = storage.dtype().clone();
        validate_storage_dtype(&dtype, &storage_dtype)?;
        Ok(Layout::from_parts(LayoutParts::new(
            ParquetVariantLayoutEncoding,
            dtype,
            storage.row_count(),
            Vec::new(),
            layout_children(vec![storage]),
            ParquetVariantLayoutData {
                storage_dtype,
                typed_paths: typed_paths.into(),
            },
        )))
    }
}

/// Checks the storage struct follows the Parquet Variant storage contract for `dtype`.
fn validate_storage_dtype(dtype: &DType, storage_dtype: &DType) -> VortexResult<()> {
    let fields = storage_dtype.as_struct_fields_opt().ok_or_else(|| {
        vortex_err!("Parquet Variant storage must be a struct, found {storage_dtype}")
    })?;
    vortex_ensure_eq!(
        storage_dtype.nullability(),
        dtype.nullability(),
        "Parquet Variant storage nullability must match the Variant dtype"
    );
    vortex_ensure!(
        fields.field("metadata").is_some_and(|f| f.is_binary()),
        "Parquet Variant storage must have a binary metadata field, found {storage_dtype}"
    );
    vortex_ensure!(
        fields.field("value").is_some() || fields.field("typed_value").is_some(),
        "Parquet Variant storage must have a value or typed_value field, found {storage_dtype}"
    );
    Ok(())
}
