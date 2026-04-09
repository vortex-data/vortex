// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::sync::Arc;

use reader::DictReader;
use vortex_array::DeserializeMetadata;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(Dict);

impl VTable for Dict {
    type Layout = DictLayout;
    type Encoding = DictLayoutEncoding;
    type Metadata = ProstMetadata<DictLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.dict")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(DictLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.codes.row_count()
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        layout.values.dtype()
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        let mut metadata =
            DictLayoutMetadata::new(PType::try_from(layout.codes.dtype()).vortex_expect("ptype"));
        metadata.is_nullable_codes = Some(layout.codes.dtype().is_nullable());
        metadata.all_values_referenced = Some(layout.all_values_referenced);
        ProstMetadata(metadata)
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        2
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match idx {
            0 => Ok(Arc::clone(&layout.values)),
            1 => Ok(Arc::clone(&layout.codes)),
            _ => vortex_bail!("Unreachable child index: {}", idx),
        }
    }

    fn child_type(_layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match idx {
            0 => LayoutChildType::Auxiliary("values".into()),
            1 => LayoutChildType::Transparent("codes".into()),
            _ => vortex_panic!("Unreachable child index: {}", idx),
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(DictReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: &ReadContext,
    ) -> VortexResult<Self::Layout> {
        let values = children.child(0, dtype)?;
        let codes_nullable = metadata
            .is_nullable_codes
            .map(Nullability::from)
            // The old behaviour (without `is_nullable_codes` metadata) used the nullability
            // of the values (and whole array).
            // see [`SerdeVTable<Dict>::build`].
            .unwrap_or_else(|| dtype.nullability());
        let codes = children.child(1, &DType::Primitive(metadata.codes_ptype(), codes_nullable))?;
        Ok(unsafe {
            DictLayout::new(values, codes)
                .set_all_values_referenced(metadata.all_values_referenced.unwrap_or(false))
        })
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 2,
            "DictLayout expects exactly 2 children (values, codes), got {}",
            children.len()
        );
        let mut children_iter = children.into_iter();
        layout.values = children_iter
            .next()
            .ok_or_else(|| vortex_err!("Missing values child"))?;
        layout.codes = children_iter
            .next()
            .ok_or_else(|| vortex_err!("Missing codes child"))?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct DictLayoutEncoding;

/// Stores a shared dictionary of values alongside compact integer codes that index into it.
///
/// Useful for columns with many repeated values, where storing each value once and
/// referencing it by index saves significant space.
#[derive(Clone, Debug)]
pub struct DictLayout {
    values: LayoutRef,
    codes: LayoutRef,
    /// Indicates whether all dictionary values are definitely referenced by at least one code.
    /// `true` = all values are referenced (computed during encoding).
    /// `false` = unknown/might have unreferenced values.
    all_values_referenced: bool,
}

impl DictLayout {
    pub(crate) fn new(values: LayoutRef, codes: LayoutRef) -> Self {
        Self {
            values,
            codes,
            all_values_referenced: false,
        }
    }

    /// Set whether all dictionary values are definitely referenced.
    ///
    /// # Safety
    /// The caller must ensure that when setting `all_values_referenced = true`, ALL dictionary
    /// values are actually referenced by at least one valid code. Setting this incorrectly can
    /// lead to incorrect query results in operations like min/max.
    ///
    /// This is typically only set to `true` during dictionary encoding when we know for certain
    /// that all values are referenced.
    /// See `DictArray::set_all_values_referenced`.
    pub unsafe fn set_all_values_referenced(mut self, all_values_referenced: bool) -> Self {
        self.all_values_referenced = all_values_referenced;
        self
    }

    pub fn has_all_values_referenced(&self) -> bool {
        self.all_values_referenced
    }
}

#[derive(prost::Message)]
pub struct DictLayoutMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    // i32 is required for proto, use the generated getter to read this field.
    codes_ptype: i32,
    // nullable codes are optional since they were added after stabilisation
    #[prost(optional, bool, tag = "2")]
    is_nullable_codes: Option<bool>,
    // all_values_referenced is optional for backward compatibility
    // true = all dictionary values are definitely referenced by at least one code
    // false/None = unknown whether all values are referenced (conservative default)
    // see `DictArray::all_values_referenced`
    #[prost(optional, bool, tag = "3")]
    pub(crate) all_values_referenced: Option<bool>,
}

impl DictLayoutMetadata {
    pub fn new(codes_ptype: PType) -> Self {
        let mut metadata = Self::default();
        metadata.set_codes_ptype(codes_ptype);
        metadata
    }
}
