// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::BoolArray;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::list_contains::ListContains as ListContainsExpr;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::validity::Validity;

/// Check list-contains without reading buffers (metadata-only).
///
/// This trait dispatches on the **element** (needle) child at index 1 of the `ListContains`
/// expression. `Self` is the concrete element encoding, while the list (haystack) is passed as an
/// opaque `&ArrayRef`; [`ListContainsListReduce`] is the mirror image, dispatching on the list.
///
/// Return `None` if the operation cannot be resolved from metadata alone.
pub trait ListContainsElementReduce: VTable {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Check list-contains, potentially reading buffers.
///
/// Like [`ListContainsElementReduce`], this dispatches on the **element** (needle) child at
/// index 1. Unlike the reduce variant, implementations may read and execute on buffers via
/// the provided [`ExecutionCtx`].
pub trait ListContainsElementKernel: VTable {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Check list-contains without reading buffers (metadata-only).
///
/// This trait dispatches on the **list** (haystack) child at index 0 of the `ListContains`
/// expression, for encodings with a specialized list representation. `Self` is the concrete list
/// encoding, while the needle is passed as an opaque `&ArrayRef`.
///
/// Return `None` if the operation cannot be resolved from metadata alone.
pub trait ListContainsListReduce: VTable {
    fn list_contains(
        list: ArrayView<'_, Self>,
        needle: &ArrayRef,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Check list-contains, potentially reading buffers.
///
/// Like [`ListContainsListReduce`], this dispatches on the **list** (haystack) child at index 0.
/// Unlike the reduce variant, implementations may read and execute on buffers via the provided
/// [`ExecutionCtx`].
pub trait ListContainsListKernel: VTable {
    fn list_contains(
        list: ArrayView<'_, Self>,
        needle: &ArrayRef,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// The constant haystack of a `list_contains`, prepared for one pass over a column of needles.
///
/// This is the `IN` shape, `list_contains(lit([...]), column)`: an element kernel builds a probe
/// structure from [`elements`](Self::elements) once and tests every needle against it, then hands
/// the membership bits back to [`finish`](Self::finish).
///
/// [`try_new`](Self::try_new) answers `None` for the shapes a kernel cannot serve — a list that is
/// not a constant, is null, or is empty, and a needle whose dtype does not match the list's
/// elements — leaving them to the generic implementation.
pub struct ListContainsSet {
    elements: Vec<Scalar>,
    nullability: Nullability,
    non_match_is_unknown: bool,
}

impl ListContainsSet {
    /// Prepares the constant list `list` to be probed by needles of dtype `needle_dtype`, or
    /// returns `None` when the kernel must decline.
    pub fn try_new(
        list: &ArrayRef,
        needle_dtype: &DType,
        options: &ListContainsOptions,
    ) -> Option<Self> {
        let DType::List(element_dtype, _) = list.dtype() else {
            return None;
        };
        // A needle of a different dtype is an invalid expression, which the generic
        // implementation reports as an error.
        if !element_dtype.eq_ignore_nullability(needle_dtype) {
            return None;
        }

        let elements = list.as_constant()?.as_list_opt()?.elements()?;
        if elements.is_empty() {
            return None;
        }

        Some(Self {
            nullability: options.result_nullability(list.dtype(), needle_dtype),
            non_match_is_unknown: options.sql_null_semantics
                && elements.iter().any(Scalar::is_null),
            elements,
        })
    }

    /// The elements of the constant list, in order, a null element included.
    ///
    /// A null element never equals a needle, so a probe structure drops it; it only decides
    /// whether a non-match is [unknown](Self::non_match_is_unknown).
    pub fn elements(&self) -> &[Scalar] {
        &self.elements
    }

    /// Whether a needle matching no element answers `null` rather than `false`.
    ///
    /// A null element under [`ListContainsOptions::sql_null_semantics`] makes the comparison
    /// unknown, the three-valued semantics of SQL `IN`.
    pub fn non_match_is_unknown(&self) -> bool {
        self.non_match_is_unknown
    }

    /// The nullability of the result, as declared by the expression's return dtype.
    pub fn nullability(&self) -> Nullability {
        self.nullability
    }

    /// Assembles the result from one membership bit per needle and the needles' validity.
    ///
    /// When a non-match is unknown, only the rows that matched stay valid.
    pub fn finish(&self, bits: BitBuffer, needle_validity: Validity) -> VortexResult<ArrayRef> {
        let validity = if self.non_match_is_unknown {
            needle_validity.and(Validity::from(bits.clone()))?
        } else {
            needle_validity
        };
        Ok(BoolArray::new(bits, validity.union_nullability(self.nullability)).into_array())
    }
}

/// Adaptor that wraps a [`ListContainsElementReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct ListContainsElementReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for ListContainsElementReduceAdaptor<V>
where
    V: ListContainsElementReduce,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the element/needle child (index 1), not the list child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let list = scalar_fn_array.get_child(0);
        <V as ListContainsElementReduce>::list_contains(list, array, parent.options)
    }
}

/// Adaptor that wraps a [`ListContainsElementKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct ListContainsElementExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for ListContainsElementExecuteAdaptor<V>
where
    V: ListContainsElementKernel,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the element/needle child (index 1), not the list child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let list = scalar_fn_array.get_child(0);
        <V as ListContainsElementKernel>::list_contains(list, array, parent.options, ctx)
    }
}

/// Adaptor that wraps a [`ListContainsListReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct ListContainsListReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for ListContainsListReduceAdaptor<V>
where
    V: ListContainsListReduce,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the list child (index 0), not the element/needle child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let needle = scalar_fn_array.get_child(1);
        <V as ListContainsListReduce>::list_contains(array, needle, parent.options)
    }
}

/// Adaptor that wraps a [`ListContainsListKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct ListContainsListExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for ListContainsListExecuteAdaptor<V>
where
    V: ListContainsListKernel,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the list child (index 0), not the element/needle child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let needle = scalar_fn_array.get_child(1);
        <V as ListContainsListKernel>::list_contains(array, needle, parent.options, ctx)
    }
}
