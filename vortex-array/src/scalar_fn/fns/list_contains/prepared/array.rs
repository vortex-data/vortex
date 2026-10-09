// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex_buffer::BitBuffer;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::Probe;
use super::new_probe;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::ArraySlots;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::constant::list_scalar_elements;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::literal::Literal;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`PreparedSet`]-encoded array.
pub type PreparedSetArray = Array<PreparedSet>;

/// A constant, non-null list whose elements are prepared as a set for membership probes.
///
/// Every row holds the same list, as in a [`ConstantArray`]. A [`PreparedSetLiteral`] applied to
/// an array, and the constant list of a [`ListContains`] node, become one when the array is
/// optimized. A [`ListContainsElementReduce`] rule of the needle encoding then pushes the function
/// into the needle's parts with slices of this array, so that the chunks of a chunked needle or the
/// values of a dictionary all probe one set, and a [`ListContainsElementKernel`] of the needle
/// encoding gets the prepared set as its list.
///
/// The set is shared, so a slice or a filter of this array does not build it again. This encoding
/// exists only in memory, and it cannot be serialized.
///
/// [`PreparedSetLiteral`]: crate::scalar_fn::fns::list_contains::PreparedSetLiteral
///
/// [`ListContains`]: crate::scalar_fn::fns::list_contains::ListContains
/// [`ListContainsElementKernel`]: crate::scalar_fn::fns::list_contains::ListContainsElementKernel
/// [`ListContainsElementReduce`]: crate::scalar_fn::fns::list_contains::ListContainsElementReduce
#[derive(Clone, Debug)]
pub struct PreparedSet;

/// The data of a [`PreparedSetArray`]: the list that every row holds, and its set.
///
/// The set is shared by every clone, so a slice or a filter of the array, and every batch a
/// [`PreparedSetLiteral`] is applied to, probe the one set. The probe needs an [`ExecutionCtx`] to
/// materialize the elements, so it is built on the first probe.
///
/// [`PreparedSetLiteral`]: crate::scalar_fn::fns::list_contains::PreparedSetLiteral
#[derive(Clone)]
pub struct PreparedSetData(Arc<SharedSet>);

struct SharedSet {
    /// The [`Literal`] of the non-null list that every row holds, shared with the expression it
    /// came from rather than cloned out of it.
    literal: ScalarFnRef,
    /// Whether the list holds a null element, which a probe does not hold.
    has_null_element: bool,
    /// Whether the list holds no element at all, counting null elements.
    is_empty: bool,
    /// The elements, and the probe over the non-null ones, built on the first probe.
    set: OnceLock<SharedVortexResult<ElementSet>>,
}

/// The elements of the list, and the non-null ones in a probe structure.
pub(super) struct ElementSet {
    /// All elements of the list, null elements included, with the element dtype of the list.
    elements: ArrayRef,
    pub(super) probe: Box<dyn Probe>,
}

impl ElementSet {
    fn try_new(list: &Scalar, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let elements = list_scalar_elements(&list.as_list(), ctx.allocator());

        // A null element never equals a needle, so the probe is better off without it.
        let valid = elements.validity()?.execute_mask(elements.len(), ctx)?;
        let probe_elements = if valid.all_true() {
            elements.clone()
        } else {
            elements.filter(valid)?
        };
        let probe = new_probe(probe_elements, ctx)?;

        Ok(Self { elements, probe })
    }
}

impl PreparedSetData {
    /// Prepares the non-null list scalar `list` as a set.
    ///
    /// # Errors
    ///
    /// Fails when `list` is not a list, or is null.
    pub fn try_new(list: Scalar) -> VortexResult<Self> {
        Self::try_from_literal(Literal.bind(list))
    }

    /// Prepares the non-null list that the [`Literal`] `literal` holds as a set, sharing the
    /// literal rather than cloning its list.
    ///
    /// # Errors
    ///
    /// Fails when `literal` is not a literal, or holds a null, or a value that is not a list.
    pub(crate) fn try_from_literal(literal: ScalarFnRef) -> VortexResult<Self> {
        let (has_null_element, is_empty) = {
            let Some(list) = literal.as_opt::<Literal>() else {
                vortex_bail!("A prepared set needs a literal list");
            };
            vortex_ensure!(
                matches!(list.dtype(), DType::List(..)),
                "A prepared set needs a list, got {}",
                list.dtype()
            );
            let Some(elements) = list.as_list().element_values() else {
                vortex_bail!("A prepared set needs a non-null list");
            };
            (elements.iter().any(Option::is_none), elements.is_empty())
        };

        Ok(Self(Arc::new(SharedSet {
            literal,
            has_null_element,
            is_empty,
            set: OnceLock::new(),
        })))
    }

    /// Whether `other` shares this set rather than preparing its own.
    #[cfg(test)]
    pub(super) fn shares_set_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Whether the list holds no element at all, null elements included.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty
    }

    /// The list that every row holds.
    pub fn list(&self) -> &Scalar {
        self.0.literal.as_::<Literal>()
    }

    /// The elements of the list, one row each, null elements included.
    ///
    /// # Errors
    ///
    /// Fails when the elements cannot be executed into the probe.
    pub fn elements(&self, ctx: &mut ExecutionCtx) -> VortexResult<&ArrayRef> {
        Ok(&self.element_set(ctx)?.elements)
    }

    /// The elements and their probe, built on the first call and shared after it.
    pub(super) fn element_set<'a>(
        &'a self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<&'a ElementSet> {
        self.0
            .set
            .get_or_init(|| ElementSet::try_new(self.list(), ctx).map_err(Arc::new))
            .as_ref()
            .map_err(|err| Arc::clone(err).into())
    }

    /// The dtype of the list that every row holds.
    fn list_dtype(&self) -> &DType {
        self.list().dtype()
    }

    /// The dtype of the list's elements.
    fn element_dtype(&self) -> &DType {
        let DType::List(element_dtype, _) = self.list_dtype() else {
            vortex_panic!("A prepared set always holds a list");
        };
        element_dtype
    }

    /// Whether each of `needles` is an element of the list, under `options`.
    ///
    /// The needles must have the dtype of the list's elements, ignoring nullability. The result
    /// has one row per needle, and the nullability that [`ListContainsOptions::result_nullability`]
    /// declares. A null needle gives `null`. The exception is an empty list off SQL null
    /// semantics, which gives `false` for every needle. Under SQL null semantics, a list that holds
    /// a null element gives `null` for a needle that matches no element.
    ///
    /// A constant needle is probed once, and gives a constant result.
    ///
    /// # Errors
    ///
    /// Fails when the needles do not have the dtype of the list's elements.
    pub fn contains(
        &self,
        needles: &ArrayRef,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if let Some(needle) = needles.as_constant() {
            let result = self.contains_scalar(&needle, options, ctx)?;
            return Ok(ConstantArray::new(result, needles.len()).into_array());
        }

        self.check_needle_dtype(needles.dtype())?;
        let (bits, needle_validity) = self.element_set(ctx)?.probe.contains(needles, ctx)?;
        self.result_from_bits(bits, needle_validity, needles.dtype(), options)
    }

    /// Whether the constant `needle` is an element of the list, under `options`.
    ///
    /// The answer is the one [`Self::contains`] gives for each row of a needle with this value, for
    /// example for the fill value of a sparse needle. The probe gets one row.
    ///
    /// # Errors
    ///
    /// Fails when the needle does not have the dtype of the list's elements.
    pub fn contains_scalar(
        &self,
        needle: &Scalar,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        self.check_needle_dtype(needle.dtype())?;
        let needles = ConstantArray::new(needle.clone(), 1).into_array();
        let (bits, needle_validity) = self.element_set(ctx)?.probe.contains(&needles, ctx)?;
        self.result_from_bits(bits, needle_validity, needle.dtype(), options)?
            .execute_scalar(0, ctx)
    }

    /// Makes the result of [`Self::contains`] from one membership bit per needle.
    ///
    /// A kernel that finds the matches of its needles without this probe, for example from bounds
    /// of its own, uses this to apply the null semantics of [`Self::contains`]. A set bit in `bits`
    /// tells that the needle equals an element. `needle_validity` is the validity of the needles,
    /// and `needle_dtype` is their dtype. The bit of a null needle has no effect.
    ///
    /// # Errors
    ///
    /// Fails when `needle_dtype` is not the dtype of the list's elements, or when
    /// `needle_validity` does not have one row per bit.
    pub fn result_from_bits(
        &self,
        bits: BitBuffer,
        needle_validity: Validity,
        needle_dtype: &DType,
        options: &ListContainsOptions,
    ) -> VortexResult<ArrayRef> {
        self.check_needle_dtype(needle_dtype)?;
        if let Some(len) = needle_validity.maybe_len() {
            vortex_ensure!(
                len == bits.len(),
                "Needle validity has {len} rows, but there are {} membership bits",
                bits.len()
            );
        }

        let nullability = options.result_nullability(self.list_dtype(), needle_dtype);

        let validity = if self.0.is_empty && !options.sql_null_semantics {
            Validity::NonNullable
        } else if options.sql_null_semantics && self.0.has_null_element {
            // Only a match is known. A comparison with the null element makes a non-match unknown.
            needle_validity.and(Validity::from(bits.clone()))?
        } else {
            needle_validity
        };

        Ok(BoolArray::new(bits, validity.union_nullability(nullability)).into_array())
    }

    /// Fails when needles of `needle_dtype` cannot be elements of the list.
    fn check_needle_dtype(&self, needle_dtype: &DType) -> VortexResult<()> {
        let element_dtype = self.element_dtype();
        if !element_dtype.eq_ignore_nullability(needle_dtype) {
            vortex_bail!(
                "Element type {} of list does not match search value {}",
                element_dtype,
                needle_dtype,
            );
        }
        Ok(())
    }
}

impl Debug for PreparedSetData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedSetData")
            .field("list", self.list())
            .finish_non_exhaustive()
    }
}

impl Display for PreparedSetData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.list())
    }
}

impl PartialEq for PreparedSetData {
    fn eq(&self, other: &Self) -> bool {
        self.list() == other.list()
    }
}

impl Eq for PreparedSetData {}

impl Hash for PreparedSetData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.list().hash(state);
    }
}

impl ArrayHash for PreparedSetData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.hash(state);
    }
}

impl ArrayEq for PreparedSetData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self == other
    }
}

impl Array<PreparedSet> {
    /// Prepares the non-null list scalar `list` as a set repeated `len` times.
    ///
    /// [`ListContains`] prepares a constant list itself. A kernel crate can use this to test its
    /// [`ListContainsElementKernel`] against a prepared set.
    ///
    /// [`ListContains`]: crate::scalar_fn::fns::list_contains::ListContains
    /// [`ListContainsElementKernel`]: crate::scalar_fn::fns::list_contains::ListContainsElementKernel
    ///
    /// # Errors
    ///
    /// Fails when `list` is not a list, or is null.
    pub fn try_new(list: Scalar, len: usize) -> VortexResult<Self> {
        Ok(Self::new(PreparedSetData::try_new(list)?, len))
    }

    /// An array of `len` rows that share the prepared set `set`.
    pub fn new(set: PreparedSetData, len: usize) -> Self {
        let dtype = set.list_dtype().clone();

        // SAFETY: the dtype is the dtype of the list that every row holds.
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(
                PreparedSet,
                dtype,
                len,
                set,
                ArraySlots::new(),
            ))
        }
    }
}

const PARENT_RULES: ParentRuleSet<PreparedSet> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(PreparedSet)),
    ParentRuleSet::lift(&SliceReduceAdaptor(PreparedSet)),
]);

impl VTable for PreparedSet {
    type TypedArrayData = PreparedSetData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.list.prepared_set");
        *ID
    }

    fn validate(
        &self,
        data: &PreparedSetData,
        dtype: &DType,
        _len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            data.list_dtype() == dtype,
            "PreparedSetArray list dtype does not match outer dtype"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("PreparedSetArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("PreparedSetArray slot_name index {idx} out of bounds")
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("PreparedSetArray is not serializable")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("PreparedSetArray is not serializable")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // The rows hold the list only. The probe is of no use to the canonical form.
        Ok(ExecutionResult::done(ConstantArray::new(
            array.data().list().clone(),
            array.len(),
        )))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<PreparedSet> for PreparedSet {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, PreparedSet>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(array.data().list().clone())
    }
}

impl ValidityVTable<PreparedSet> for PreparedSet {
    fn validity(_array: ArrayView<'_, PreparedSet>) -> VortexResult<Validity> {
        // The list is never null.
        Ok(Validity::AllValid)
    }
}

impl SliceReduce for PreparedSet {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            PreparedSetArray::new(array.data().clone(), range.len()).into_array(),
        ))
    }
}

impl FilterReduce for PreparedSet {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            PreparedSetArray::new(array.data().clone(), mask.true_count()).into_array(),
        ))
    }
}
