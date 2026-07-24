// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::sync::Arc;

use reader::StructReader;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::SessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
pub use writer::StructStrategy;

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

/// Struct layout vtable.
#[derive(Clone, Debug)]
pub struct Struct;

/// Backwards-compatible name for the struct layout plugin.
pub use Struct as StructLayoutEncoding;

/// A layout decomposing a struct into one child per field and optional validity.
pub type StructLayout = Layout<Struct>;

impl VTable for Struct {
    type LayoutData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.struct");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        Layout::<Struct>::validate_children(args.dtype, args.children.nchildren())?;

        for idx in 0..args.children.nchildren() {
            let child_row_count = args.children.child_row_count(idx);
            vortex_ensure!(
                child_row_count == args.row_count,
                "Struct child {idx} row count does not match parent"
            );
        }
        Ok(())
    }

    fn child_dtype(layout: &Layout<Self>, index: usize) -> VortexResult<DType> {
        StructLayout::child_dtype(layout.dtype(), index)
    }

    fn child_type(layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        let schema_index = if layout.dtype().is_nullable() {
            idx.saturating_sub(1)
        } else {
            idx
        };
        if idx == 0 && layout.dtype().is_nullable() {
            LayoutChildType::Auxiliary("validity".into())
        } else {
            LayoutChildType::Field(
                layout
                    .struct_fields()
                    .field_name(schema_index)
                    .vortex_expect("Field index out of bounds")
                    .clone(),
            )
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(StructReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.session(),
            ctx.clone(),
        )?))
    }
}

impl Layout<Struct> {
    /// Construct a struct layout from owned children.
    pub fn new(row_count: u64, dtype: DType, children: Vec<LayoutRef>) -> Self {
        Self::validate_children(&dtype, children.len()).vortex_expect("invalid struct children");
        LayoutParts::new(
            Struct,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(children),
            (),
        )
        .into_typed()
    }

    /// Returns the struct fields.
    pub fn struct_fields(&self) -> &StructFields {
        self.dtype()
            .as_struct_fields_opt()
            .vortex_expect("Struct layout dtype must be a struct")
    }

    /// Invokes `per_child` for fields selected by `field_mask`.
    pub fn matching_fields<F>(&self, field_mask: &[FieldMask], mut per_child: F) -> VortexResult<()>
    where
        F: FnMut(FieldMask, usize) -> VortexResult<()>,
    {
        if field_mask.iter().any(|mask| mask.matches_all()) {
            for idx in 0..self.struct_fields().nfields() {
                per_child(FieldMask::All, idx)?;
            }
            return Ok(());
        }

        for path in field_mask {
            let Some(field) = path.starting_field()? else {
                continue;
            };
            let Field::Name(field_name) = field else {
                vortex_bail!("Expected field name, got {field:?}");
            };
            let idx = self
                .struct_fields()
                .find(field_name)
                .ok_or_else(|| vortex_err!("Field not found: {field_name}"))?;
            per_child(path.clone().step_into()?, idx)?;
        }
        Ok(())
    }

    fn validate_children(dtype: &DType, nchildren: usize) -> VortexResult<()> {
        let fields = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Expected struct dtype"))?;
        let expected = fields.nfields() + usize::from(dtype.is_nullable());
        vortex_ensure!(
            nchildren == expected,
            "Struct layout has {nchildren} children, expected {expected}"
        );
        Ok(())
    }

    fn child_dtype(dtype: &DType, index: usize) -> VortexResult<DType> {
        let schema_index = if dtype.is_nullable() {
            index.saturating_sub(1)
        } else {
            index
        };
        if index == 0 && dtype.is_nullable() {
            Ok(DType::Bool(Nullability::NonNullable))
        } else {
            dtype
                .as_struct_fields_opt()
                .and_then(|fields| fields.field_by_index(schema_index))
                .ok_or_else(|| vortex_err!("Missing field {schema_index}"))
        }
    }
}
