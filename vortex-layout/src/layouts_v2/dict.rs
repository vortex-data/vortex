// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::LayoutChildType;
use crate::LayoutId;
use crate::layout_v2::Layout;
use crate::layout_v2::LayoutDeserializeArgs;
use crate::layout_v2::VTable;
use crate::layout_v2::metadata_bool_field;
use crate::layout_v2::metadata_varint_field;
use crate::scan::plan::ScanPlanRef;
use crate::scan::plan::request::ScanRequest;
use crate::scan::v2::layouts::dict as scan_dict;

/// V2 dictionary layout vtable.
#[derive(Clone, Debug)]
pub struct Dict;

/// V2 dictionary layout data.
#[derive(Clone, Debug)]
pub struct DictData {
    pub(crate) codes_dtype: DType,
    pub(crate) all_values_referenced: bool,
}

impl DictData {
    /// Returns whether all dictionary values are definitely referenced.
    pub fn has_all_values_referenced(&self) -> bool {
        self.all_values_referenced
    }
}

impl VTable for Dict {
    type LayoutData = DictData;

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.dict")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        let codes_ptype = metadata_varint_field(args.metadata, 1)?
            .ok_or_else(|| vortex_err!("Dict metadata missing codes ptype"))?;
        let codes_ptype = PType::try_from(i32::try_from(codes_ptype)?)?;
        let codes_nullable = metadata_bool_field(args.metadata, 2)?
            .map(Nullability::from)
            .unwrap_or_else(|| args.dtype.nullability());
        Ok(DictData {
            codes_dtype: DType::Primitive(codes_ptype, codes_nullable),
            all_values_referenced: metadata_bool_field(args.metadata, 3)?.unwrap_or(false),
        })
    }

    fn child_dtype(layout: Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            0 => Ok(layout.dtype().clone()),
            1 => Ok(layout.data().codes_dtype.clone()),
            _ => vortex_bail!("Dict child index out of bounds: {idx}"),
        }
    }

    fn child_type(_layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        match idx {
            0 => Ok(LayoutChildType::Auxiliary("values".into())),
            1 => Ok(LayoutChildType::Transparent("codes".into())),
            _ => vortex_bail!("Dict child index out of bounds: {idx}"),
        }
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        scan_dict::new_scan_plan(layout, req, session)
    }
}
