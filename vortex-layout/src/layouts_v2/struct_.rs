// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::DeserializeMetadata;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::LayoutChildType;
use crate::LayoutId;
use crate::layout_v2::Layout;
use crate::layout_v2::LayoutDeserializeArgs;
use crate::layout_v2::VTable;
use crate::scan::plan::ScanPlanRef;
use crate::scan::plan::request::ScanRequest;
use crate::scan::v2::layouts::struct_ as scan_struct;

/// V2 struct layout vtable.
#[derive(Clone, Debug)]
pub struct Struct;

impl VTable for Struct {
    type LayoutData = ();

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.struct")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        EmptyMetadata::deserialize(args.metadata)?;
        Ok(())
    }

    fn child_dtype(layout: Layout<Self>, idx: usize) -> VortexResult<DType> {
        let schema_index = if layout.dtype().is_nullable() {
            idx.saturating_sub(1)
        } else {
            idx
        };
        if idx == 0 && layout.dtype().is_nullable() {
            Ok(DType::Bool(Nullability::NonNullable))
        } else {
            layout
                .dtype()
                .as_struct_fields_opt()
                .and_then(|fields| fields.field_by_index(schema_index))
                .ok_or_else(|| vortex_err!("Missing struct field {schema_index}"))
        }
    }

    fn child_type(layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        let schema_index = if layout.dtype().is_nullable() {
            idx.saturating_sub(1)
        } else {
            idx
        };
        if idx == 0 && layout.dtype().is_nullable() {
            Ok(LayoutChildType::Auxiliary("validity".into()))
        } else {
            let name = layout
                .dtype()
                .as_struct_fields_opt()
                .and_then(|fields| fields.field_name(schema_index))
                .ok_or_else(|| vortex_err!("Missing struct field {schema_index}"))?;
            Ok(LayoutChildType::Field(name.clone()))
        }
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        scan_struct::new_scan_plan(layout, req, session)
    }
}
