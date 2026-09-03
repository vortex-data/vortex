// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanChildren;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;

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

    fn execute(
        plan: &Plan<Self>,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_ensure!(
            row_range.start <= row_range.end && row_range.end <= plan.row_count(),
            "ListPack row range {:?} is outside 0..{}",
            row_range,
            plan.row_count()
        );
        let row_count = usize::try_from(row_range.end - row_range.start)?;
        vortex_ensure!(mask.len() == row_count, "ListPack mask length mismatch");

        let offsets_range = row_range.start
            ..row_range
                .end
                .checked_add(1)
                .ok_or_else(|| vortex_err!("List offsets range overflow"))?;
        let offsets = plan.offsets()?.execute(
            ctx,
            &offsets_range,
            MaskFuture::new_true(row_count.saturating_add(1)),
        )?;
        let validity = plan
            .validity()?
            .map(|validity| validity.execute(ctx, row_range, MaskFuture::new_true(row_count)))
            .transpose()?;
        let elements = plan.elements()?;
        let execution = ctx.clone();
        let dtype = plan.dtype().clone();
        let nullability = dtype.nullability();

        Ok(async move {
            let (offsets, mask) = try_join!(offsets, mask)?;
            if mask.all_false() {
                return Ok(Canonical::empty(&dtype).into_array());
            }

            let elements_range = elements_range_from_offsets(&offsets, execution.session())?;
            let elements_count = usize::try_from(elements_range.end - elements_range.start)?;
            let elements = elements
                .execute(
                    &execution,
                    &elements_range,
                    MaskFuture::new_true(elements_count),
                )?
                .await?;
            let validity = match validity {
                Some(validity) => Some(validity.await?),
                None => None,
            };
            let offsets = rebase_offsets(offsets, elements_range.start)?;
            // SAFETY: lowering from a list layout guarantees compatible elements and monotonically
            // increasing offsets. Rebasing preserves the represented list lengths.
            let list = unsafe {
                ListArray::new_unchecked(elements, offsets, create_validity(validity, nullability))
            }
            .into_array();
            if mask.all_true() {
                Ok(list)
            } else {
                list.filter(mask)
            }
        }
        .boxed())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match index {
            ELEMENTS => Cow::Borrowed("elements"),
            OFFSETS => Cow::Borrowed("offsets"),
            VALIDITY => Cow::Borrowed("validity"),
            _ => Cow::Owned(format!("child[{index}]")),
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

fn elements_range_from_offsets(
    offsets: &ArrayRef,
    session: &vortex_session::VortexSession,
) -> VortexResult<Range<u64>> {
    if offsets.is_empty() {
        return Ok(0..0);
    }
    let mut ctx = session.create_execution_ctx();
    let start = offsets
        .execute_scalar(0, &mut ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("offset value must fit in u64");
    let end = offsets
        .execute_scalar(offsets.len() - 1, &mut ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("offset value must fit in u64");
    Ok(start..end)
}

fn rebase_offsets(offsets: ArrayRef, first: u64) -> VortexResult<ArrayRef> {
    if first == 0 {
        return Ok(offsets);
    }
    let constant = ConstantArray::new(first, offsets.len())
        .into_array()
        .cast(offsets.dtype().clone())?;
    offsets.binary(constant, Operator::Sub)
}

fn create_validity(validity: Option<ArrayRef>, nullability: Nullability) -> Validity {
    match validity {
        Some(validity) => Validity::Array(validity),
        None if nullability.is_nullable() => Validity::AllValid,
        None => Validity::NonNullable,
    }
}
