// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;

use std::sync::Arc;

use reader::StructReader;
use vortex_array::DeserializeMetadata;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::SessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PlanArguments;
use crate::v2::project::ProjectPlan;
use crate::v2::struct_::StructPlan;
use crate::vtable;

vtable!(Struct);

impl VTable for Struct {
    type Layout = StructLayout;
    type Encoding = StructLayoutEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new("vortex.struct")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(StructLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(_layout: &Self::Layout) -> Self::Metadata {
        EmptyMetadata
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        let validity_children = if layout.dtype.is_nullable() { 1 } else { 0 };
        layout.struct_fields().nfields() + validity_children
    }

    fn child(layout: &Self::Layout, index: usize) -> VortexResult<LayoutRef> {
        let schema_index = if layout.dtype.is_nullable() {
            index.saturating_sub(1)
        } else {
            index
        };

        let child_dtype = if index == 0 && layout.dtype.is_nullable() {
            DType::Bool(Nullability::NonNullable)
        } else {
            layout
                .struct_fields()
                .field_by_index(schema_index)
                .ok_or_else(|| vortex_err!("Missing field {schema_index}"))?
        };

        layout.children.child(index, &child_dtype)
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        let schema_index = if layout.dtype.is_nullable() {
            idx.saturating_sub(1)
        } else {
            idx
        };

        if idx == 0 && layout.dtype.is_nullable() {
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
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(StructReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.session(),
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: &ReadContext,
    ) -> VortexResult<Self::Layout> {
        let struct_dt = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Expected struct dtype"))?;

        let expected_children = struct_dt.nfields() + (dtype.is_nullable() as usize);
        vortex_ensure!(
            children.nchildren() == expected_children,
            "Struct layout has {} children, but dtype has {} fields",
            children.nchildren(),
            struct_dt.nfields()
        );

        Ok(StructLayout {
            row_count,
            dtype: dtype.clone(),
            children: children.to_arc(),
        })
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        let struct_dt = layout
            .dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Expected struct dtype"))?;

        let expected_children = struct_dt.nfields() + (layout.dtype.is_nullable() as usize);
        vortex_ensure!(
            children.len() == expected_children,
            "StructLayout expects {} children, got {}",
            expected_children,
            children.len()
        );

        layout.children = OwnedLayoutChildren::layout_children(children);
        Ok(())
    }

    fn plan(layout: &Self::Layout, args: PlanArguments) -> VortexResult<LayoutPlanRef> {
        // The full design (`LAYOUT_PLAN.md` § StructLayout::plan) routes
        // each referenced field to its own child plan and re-assembles
        // via the original expression at the top. We use the
        // production-tested `partition()` infrastructure for that
        // (see `vortex_layout::layouts::struct_::reader::StructReader`):
        //
        // 1. Expand `root()` to an explicit `pack` of all fields.
        // 2. Partition the expression into per-field sub-expressions
        //    plus a re-assembly root that sees the partition results.
        // 3. Step into each field's scope (rewrite `col(name)` -> `root()`).
        // 4. Recurse into the field's child layout — only the touched
        //    fields are read.
        // 5. `StructPlan` zips the partition results into a struct
        //    keyed by partition name, then `ProjectPlan` evaluates the
        //    re-assembly root if it isn't already a no-op.
        //
        // Nullable structs need the validity child threaded through —
        // wired in alongside `FilterPlan`.
        if layout.dtype.is_nullable() {
            vortex_bail!("StructLayout::plan does not yet support nullable structs");
        }

        let struct_fields = layout.struct_fields();

        // Fast path: a `root()` projection with no expression. Read
        // every field with `root()` and zip back into the layout's
        // original struct dtype. Avoids running the partition pipeline
        // when there's nothing to rewrite.
        if vortex_array::expr::is_root(&args.expr) {
            return plan_full_struct(layout, struct_fields, args);
        }

        let expanded =
            vortex_array::expr::transform::replace_root_fields(args.expr.clone(), struct_fields);
        let expanded = expanded.optimize_recursive(&layout.dtype)?;

        let vortex_array::expr::transform::PartitionedExpr {
            root,
            partitions,
            partition_names,
            partition_dtypes,
            ..
        } = vortex_array::expr::transform::partition(
            expanded,
            &layout.dtype,
            vortex_array::expr::analysis::make_free_field_annotator(struct_fields),
        )?;

        if partitions.is_empty() {
            // Expression doesn't reference any fields (e.g., `lit(1)`).
            // Fall back to reading every field, then evaluating on top.
            return plan_full_struct_with_projection(layout, struct_fields, args);
        }

        let mut child_plans = Vec::with_capacity(partitions.len());
        let mut intermediate_dtypes = Vec::with_capacity(partitions.len());
        let mut intermediate_names = Vec::with_capacity(partitions.len());

        for (partition, partition_name) in partitions.iter().zip(partition_names.iter()) {
            let field_idx = struct_fields
                .find(partition_name)
                .ok_or_else(|| vortex_err!("Unknown field name in partition: {partition_name}"))?;
            let child_dtype = struct_fields
                .field_by_index(field_idx)
                .ok_or_else(|| vortex_err!("Missing struct field at index {field_idx}"))?;

            // Step into the field's scope: the partition expression
            // currently does `col(name)`/`get_item(name, root())` to
            // reach into the parent struct, but the child plan
            // evaluates against the field's data directly.
            let stepped = vortex_array::expr::transform::replace(
                partition.clone(),
                &vortex_array::expr::col(partition_name.clone()),
                vortex_array::expr::root(),
            );
            let stepped = stepped.optimize_recursive(&child_dtype)?;
            let intermediate_dtype = stepped.return_dtype(&child_dtype)?;

            let child = layout.children.child(field_idx, &child_dtype)?;
            child_plans.push(child.plan(args.clone().with_expr(stepped))?);
            intermediate_dtypes.push(intermediate_dtype);
            intermediate_names.push(partition_name.clone());
        }

        let intermediate_dtype = DType::Struct(
            StructFields::new(intermediate_names.clone().into(), intermediate_dtypes),
            Nullability::NonNullable,
        );
        let struct_plan: LayoutPlanRef = Arc::new(StructPlan::new(
            child_plans,
            intermediate_names,
            intermediate_dtype.clone(),
            layout.row_count,
        ));

        // If the re-assembly root is just `root()`, the partition
        // outputs already match the requested dtype. partition_dtypes
        // is only consumed in the projection branch below.
        if vortex_array::expr::is_root(&root) {
            drop(partition_dtypes);
            return Ok(struct_plan);
        }

        let output_dtype = root.return_dtype(&intermediate_dtype)?;
        Ok(Arc::new(ProjectPlan::new(struct_plan, root, output_dtype)))
    }
}

/// Plan every field with `root()` and zip the children into the
/// layout's struct dtype with no top-level projection.
fn plan_full_struct(
    layout: &StructLayout,
    struct_fields: &StructFields,
    args: PlanArguments,
) -> VortexResult<LayoutPlanRef> {
    let nfields = struct_fields.nfields();
    let mut child_plans = Vec::with_capacity(nfields);
    let mut field_names: Vec<FieldName> = Vec::with_capacity(nfields);
    let child_args = args.with_expr(vortex_array::expr::root());
    for idx in 0..nfields {
        let child_dtype = struct_fields
            .field_by_index(idx)
            .ok_or_else(|| vortex_err!("Missing struct field at index {idx}"))?;
        let field_name = struct_fields
            .field_name(idx)
            .ok_or_else(|| vortex_err!("Missing struct field name at index {idx}"))?
            .clone();
        let child = layout.children.child(idx, &child_dtype)?;
        child_plans.push(child.plan(child_args.clone())?);
        field_names.push(field_name);
    }
    Ok(Arc::new(StructPlan::new(
        child_plans,
        field_names,
        layout.dtype.clone(),
        layout.row_count,
    )))
}

/// Same as `plan_full_struct` but additionally wraps the result with a
/// top-level `ProjectPlan` evaluating the original expression. Used for
/// expressions that don't actually reference any field (e.g. `lit(1)`).
fn plan_full_struct_with_projection(
    layout: &StructLayout,
    struct_fields: &StructFields,
    args: PlanArguments,
) -> VortexResult<LayoutPlanRef> {
    let output_dtype = args.expr.return_dtype(&layout.dtype)?;
    let expr = args.expr.clone();
    let inner = plan_full_struct(layout, struct_fields, args)?;
    Ok(Arc::new(ProjectPlan::new(inner, expr, output_dtype)))
}

#[derive(Debug)]
pub struct StructLayoutEncoding;

/// Decomposes a struct-typed column into one child per field, enabling columnar projection.
///
/// Queries that only need a subset of fields can skip reading the rest entirely.
#[derive(Clone, Debug)]
pub struct StructLayout {
    row_count: u64,
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
}

impl StructLayout {
    pub fn new(row_count: u64, dtype: DType, children: Vec<LayoutRef>) -> Self {
        Self {
            row_count,
            dtype,
            children: OwnedLayoutChildren::layout_children(children),
        }
    }

    pub fn struct_fields(&self) -> &StructFields {
        self.dtype
            .as_struct_fields_opt()
            .vortex_expect("Struct layout dtype must be a struct")
    }

    #[inline]
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    #[inline]
    pub fn children(&self) -> &Arc<dyn LayoutChildren> {
        &self.children
    }

    pub fn matching_fields<F>(&self, field_mask: &[FieldMask], mut per_child: F) -> VortexResult<()>
    where
        F: FnMut(FieldMask, usize) -> VortexResult<()>,
    {
        // If the field mask contains an `All` fields, then enumerate all fields.
        if field_mask.iter().any(|mask| mask.matches_all()) {
            for idx in 0..self.struct_fields().nfields() {
                per_child(FieldMask::All, idx)?;
            }
            return Ok(());
        }

        // Enumerate each field in the mask
        for path in field_mask {
            let Some(field) = path.starting_field()? else {
                // skip fields not in mask
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
}
