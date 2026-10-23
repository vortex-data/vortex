// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::sync::Arc;

use reader::DictReader;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentSource;

/// Dictionary layout vtable.
#[derive(Clone, Debug)]
pub struct Dict;

/// Backwards-compatible name for the dictionary layout plugin.
pub use Dict as DictLayoutEncoding;

/// Dictionary-layout-specific data.
#[derive(Clone, Debug)]
pub struct DictData {
    codes_dtype: DType,
    all_values_referenced: bool,
}

/// A dictionary values child paired with a codes child.
pub type DictLayout = Layout<Dict>;

impl VTable for Dict {
    type LayoutData = DictData;
    type Metadata = ProstMetadata<DictLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.dict");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        let mut metadata = DictLayoutMetadata::new(
            PType::try_from(&layout.codes_dtype).vortex_expect("codes ptype"),
        );
        metadata.is_nullable_codes = Some(layout.codes_dtype.is_nullable());
        metadata.all_values_referenced = Some(layout.all_values_referenced);
        ProstMetadata(metadata)
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &DictLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        let codes_ptype = metadata.get_codes_ptype()?;
        vortex_ensure_eq!(
            args.children.nchildren(),
            2,
            "DictLayout expects exactly 2 children"
        );
        let codes_nullable = metadata
            .is_nullable_codes
            .map(Nullability::from)
            .unwrap_or_else(|| args.dtype.nullability());
        let codes_dtype = DType::Primitive(codes_ptype, codes_nullable);
        args.children.child(0, args.dtype)?;
        let codes = args.children.child(1, &codes_dtype)?;
        vortex_ensure_eq!(
            codes.row_count(),
            args.row_count,
            "Dictionary codes row count does not match parent"
        );
        Ok(DictData {
            codes_dtype,
            all_values_referenced: metadata.all_values_referenced.unwrap_or(false),
        })
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            0 => Ok(layout.dtype().clone()),
            1 => Ok(layout.codes_dtype.clone()),
            _ => vortex_bail!("Dict child index out of bounds: {idx}"),
        }
    }

    fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        match idx {
            0 => LayoutChildType::Auxiliary("values".into()),
            1 => LayoutChildType::Transparent("codes".into()),
            _ => vortex_panic!("Dict child index out of bounds: {idx}"),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(DictReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx.clone(),
        )?))
    }
}

impl Layout<Dict> {
    pub(crate) fn new(values: LayoutRef, codes: LayoutRef) -> Self {
        Self::new_with_all_values_referenced(values, codes, false)
    }

    fn new_with_all_values_referenced(
        values: LayoutRef,
        codes: LayoutRef,
        all_values_referenced: bool,
    ) -> Self {
        let dtype = values.dtype().clone();
        let row_count = codes.row_count();
        let codes_dtype = codes.dtype().clone();
        LayoutParts::new(
            Dict,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(vec![values, codes]),
            DictData {
                codes_dtype,
                all_values_referenced,
            },
        )
        .into_typed()
    }

    /// Returns whether every dictionary value is known to be referenced.
    pub fn has_all_values_referenced(&self) -> bool {
        self.all_values_referenced
    }
}

#[derive(prost::Message)]
pub struct DictLayoutMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    codes_ptype: i32,
    #[prost(optional, bool, tag = "2")]
    is_nullable_codes: Option<bool>,
    #[prost(optional, bool, tag = "3")]
    pub(crate) all_values_referenced: Option<bool>,
}

impl DictLayoutMetadata {
    pub fn new(codes_ptype: PType) -> Self {
        let mut metadata = Self::default();
        metadata.set_codes_ptype(codes_ptype);
        metadata
    }

    fn get_codes_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.codes_ptype).map_err(|_| {
            vortex_err!(
                "DictLayoutMetadata codes_ptype must be a known PType, got {}",
                self.codes_ptype
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::Dict;
    use super::DictLayoutMetadata;
    use crate::LayoutBuildContext;
    use crate::LayoutVTablePlugin;
    use crate::layout_children;
    use crate::test::SESSION;

    #[test]
    fn test_build_rejects_unknown_codes_ptype() -> VortexResult<()> {
        let metadata = DictLayoutMetadata {
            codes_ptype: 11,
            ..Default::default()
        }
        .encode_to_vec();
        let children = layout_children(vec![]);
        let read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session: &SESSION,
            array_read_ctx: &read_ctx,
        };
        let result = LayoutVTablePlugin::build(
            &Dict,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            0,
            &metadata,
            vec![],
            children.as_ref(),
            &build_ctx,
        );
        assert!(result.is_err_and(|err| {
            err.to_string()
                .contains("DictLayoutMetadata codes_ptype must be a known PType, got 11")
        }));

        assert_eq!(
            DictLayoutMetadata::new(PType::U32).get_codes_ptype()?,
            PType::U32
        );

        Ok(())
    }
}
