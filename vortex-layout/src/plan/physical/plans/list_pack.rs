// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use crate::plan::physical::Plan;
use crate::plan::physical::PlanChildren;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::operators::ListSource;
use crate::plan::physical::operators::UNBOUNDED;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;
use crate::plan::physical::pipeline::Reach;
use crate::segments::SegmentId;

const ELEMENTS: usize = 0;
const OFFSETS: usize = 1;
const VALIDITY: usize = 2;

/// Assembles a list from elements and offsets, plus an optional trailing validity child.
#[derive(Clone, Debug)]
pub struct ListPack;

/// Operator-specific list assembly data.
#[derive(Clone, Debug)]
pub struct ListPackData;

/// A plan that assembles a list from its children.
pub type ListPackPlan = Plan<ListPack>;

impl ListPackPlan {
    /// Creates a list assembly from potentially unresolved children without validation.
    ///
    /// # Safety
    ///
    /// `dtype` must be a list whose element dtype matches the elements child. The offsets child
    /// must be a non-nullable integer with `row_count + 1` rows. A non-nullable boolean validity
    /// child with `row_count` rows must be present exactly when `dtype` is nullable.
    pub(crate) unsafe fn from_children_unchecked(
        dtype: DType,
        row_count: u64,
        children: PlanChildren,
    ) -> Self {
        PlanParts {
            vtable: ListPack,
            dtype,
            row_count,
            children,
            data: ListPackData,
        }
        .into_typed()
    }

    /// Creates a list assembly from `elements` and `offsets`.
    ///
    /// `validity` is required exactly when `nullability` is [`Nullability::Nullable`]. The row
    /// domain is one fewer than the number of offsets.
    pub fn try_new(
        nullability: Nullability,
        row_count: u64,
        elements: PlanRef,
        offsets: PlanRef,
        validity: Option<PlanRef>,
    ) -> VortexResult<Self> {
        let dtype = DType::List(Arc::new(elements.dtype().clone()), nullability);
        let mut children = vec![elements, offsets];
        children.extend(validity);
        let children = PlanChildren::from(children);
        validate_children(&dtype, row_count, &children)?;

        // SAFETY: All child shape invariants were validated above.
        Ok(unsafe { Self::from_children_unchecked(dtype, row_count, children) })
    }

    /// Returns the plan producing list elements.
    pub fn elements(&self) -> VortexResult<PlanRef> {
        self.child_required(ELEMENTS)
    }

    /// Returns the plan producing list offsets.
    pub fn offsets(&self) -> VortexResult<PlanRef> {
        self.child_required(OFFSETS)
    }

    /// Returns the plan producing list validity, if the list is nullable.
    pub fn validity(&self) -> VortexResult<Option<PlanRef>> {
        self.child(VALIDITY)
    }
}

impl PlanVTable for ListPack {
    type PlanData = ListPackData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.list_pack");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // Nullability is recoverable from the plan dtype.
        Some(EmptyMetadata)
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        validate_children(plan.dtype(), plan.row_count(), children)
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match index {
            ELEMENTS => Cow::Borrowed("elements"),
            OFFSETS => Cow::Borrowed("offsets"),
            VALIDITY => Cow::Borrowed("validity"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        if mask.all_false() {
            return Ok(None);
        }
        let boundaries = offset_mask(mask);
        let offsets = compiler
            .compile(&plan.offsets()?, rows.start..rows.end + 1, &boundaries)?
            .ok_or_else(|| vortex_err!("Selected list offsets produced no rows"))?;
        let mut chains = vec![offsets];
        if let Some(validity) = plan.validity()? {
            chains.push(
                compiler
                    .compile(&validity, rows, mask)?
                    .ok_or_else(|| vortex_err!("Selected list validity produced no rows"))?,
            );
        }
        let source = ListSource::new(plan.clone(), mask.clone(), compiler.values().clone());
        Ok(Some(compiler.join_with(chains, |_| UNBOUNDED, source)))
    }

    fn reach(
        plan: &Plan<Self>,
        rows: Range<u64>,
        at: &Reach,
        visit: &mut dyn FnMut(SegmentId, Range<u64>),
    ) -> VortexResult<()> {
        // Elements have a different row domain, known only after the selected offsets arrive.
        // Their scan records its own segment readers when those ranges have been resolved.
        let root_rows = at.root(&rows);
        plan.offsets()?
            .reach(rows.start..rows.end + 1, &at.fixed(&rows), &mut |id, _| {
                visit(id, root_rows.clone())
            })?;
        if let Some(validity) = plan.validity()? {
            validity.reach(rows, at, visit)?;
        }
        Ok(())
    }
}

/// A selected list needs both its starting and ending offsets.
fn offset_mask(mask: &Mask) -> Mask {
    match mask.indices() {
        AllOr::None => Mask::new_false(mask.len() + 1),
        AllOr::All => Mask::new_true(mask.len() + 1),
        AllOr::Some(indices) => {
            let mut boundaries = Vec::with_capacity(indices.len() * 2);
            for &index in indices {
                if boundaries.last() != Some(&index) {
                    boundaries.push(index);
                }
                boundaries.push(index + 1);
            }
            Mask::from_indices(mask.len() + 1, boundaries)
        }
    }
}

fn validate_children(dtype: &DType, row_count: u64, children: &PlanChildren) -> VortexResult<()> {
    let elements_dtype = dtype
        .as_list_element_opt()
        .ok_or_else(|| vortex_err!("ListPack output dtype must be a list, got {dtype}"))?;
    let expected_children = 2 + usize::from(dtype.is_nullable());
    if children.len() != expected_children {
        vortex_bail!(
            "ListPack expects {expected_children} children but got {}",
            children.len()
        );
    }

    let elements = children
        .get(ELEMENTS)?
        .ok_or_else(|| vortex_err!("ListPack elements child is absent"))?;
    if elements.dtype() != elements_dtype.as_ref() {
        vortex_bail!(
            "ListPack elements child has dtype {} but the list element dtype is {}",
            elements.dtype(),
            elements_dtype
        );
    }

    let offsets = children
        .get(OFFSETS)?
        .ok_or_else(|| vortex_err!("ListPack offsets child is absent"))?;
    if !offsets.dtype().is_int() || offsets.dtype().is_nullable() {
        vortex_bail!(
            "ListPack offsets child must have a non-nullable integer dtype, got {}",
            offsets.dtype()
        );
    }
    let offsets_row_count = row_count
        .checked_add(1)
        .ok_or_else(|| vortex_err!("ListPack offsets row count overflow"))?;
    if offsets.row_count() != offsets_row_count {
        vortex_bail!(
            "ListPack offsets child has {} rows but must have {offsets_row_count}",
            offsets.row_count()
        );
    }

    if dtype.is_nullable() {
        let validity = children
            .get(VALIDITY)?
            .ok_or_else(|| vortex_err!("ListPack validity child is absent"))?;
        let validity_dtype = DType::Bool(Nullability::NonNullable);
        if validity.dtype() != &validity_dtype {
            vortex_bail!(
                "ListPack validity child has dtype {} but must have dtype {validity_dtype}",
                validity.dtype()
            );
        }
        if validity.row_count() != row_count {
            vortex_bail!(
                "ListPack validity child has {} rows but the plan has {row_count}",
                validity.row_count()
            );
        }
    }
    Ok(())
}
