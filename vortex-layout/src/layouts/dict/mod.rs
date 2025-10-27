// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::sync::Arc;

use reader::DictReader;
use vortex_array::{ArrayContext, DeserializeMetadata, ProstMetadata};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};

use crate::children::LayoutChildren;
use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildType, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef, VTable, vtable,
};

vtable!(Dict);

impl VTable for DictVTable {
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
            0 => Ok(layout.values.clone()),
            1 => Ok(layout.codes.clone()),
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
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(DictReader::try_new(
            layout.clone(),
            name,
            segment_source,
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: ArrayContext,
    ) -> VortexResult<Self::Layout> {
        let values = children.child(0, dtype)?;
        let codes_nullable = metadata
            .is_nullable_codes
            .map(Nullability::from)
            // The old behaviour (without `is_nullable_codes` metadata) used the nullability
            // of the values (and whole array).
            // see [`SerdeVTable<DictVTable>::build`].
            .unwrap_or_else(|| dtype.nullability());
        let codes = children.child(1, &DType::Primitive(metadata.codes_ptype(), codes_nullable))?;
        Ok(DictLayout { values, codes })
    }
}

#[derive(Debug)]
pub struct DictLayoutEncoding;

#[derive(Clone, Debug)]
pub struct DictLayout {
    values: LayoutRef,
    codes: LayoutRef,
}

impl DictLayout {
    pub(super) fn new(values: LayoutRef, codes: LayoutRef) -> Self {
        Self { values, codes }
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
}

impl DictLayoutMetadata {
    pub fn new(codes_ptype: PType) -> Self {
        let mut metadata = Self::default();
        metadata.set_codes_ptype(codes_ptype);
        metadata
    }
}
