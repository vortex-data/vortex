// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
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
use crate::arrays::ListViewArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::match_smallest_list_offset_type;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`PreparedSet`]-encoded array.
pub type PreparedSetArray = Array<PreparedSet>;

/// A constant, non-null list whose elements are prepared as a set for membership probes.
///
/// Every row holds the same list, as in a [`ConstantArray`]. When [`ListContains`] gets a constant
/// list and a needle that is not canonical, it puts this array in place of the list and gives the
/// node back to the executor. Thus a [`ListContainsElementKernel`] of the needle encoding gets the
/// prepared set as its list, and can probe its own values with [`PreparedSetData::contains`].
///
/// The probe is shared, so a slice or a filter of this array does not build it again. This encoding
/// exists only during execution, and it cannot be serialized.
///
/// [`ListContains`]: crate::scalar_fn::fns::list_contains::ListContains
/// [`ListContainsElementKernel`]: crate::scalar_fn::fns::list_contains::ListContainsElementKernel
#[derive(Clone, Debug)]
pub struct PreparedSet;

/// The data of a [`PreparedSetArray`]: the elements of the list, and the probe built from them.
#[derive(Clone)]
pub struct PreparedSetData {
    /// All elements of the list, null elements included, with the element dtype of the list.
    elements: ArrayRef,
    /// The nullability of the list dtype. The list itself is never null.
    nullability: Nullability,
    pub(super) set: Arc<ElementSet>,
}

/// The non-null elements of the list in a probe structure, and the facts about the list that
/// decide the answer when no element matches.
pub(super) struct ElementSet {
    pub(super) probe: Box<dyn Probe>,
    /// Whether the list holds a null element, which a probe does not hold.
    has_null_element: bool,
    /// Whether the list holds no element at all, counting null elements.
    is_empty: bool,
}

impl PreparedSetData {
    /// Prepares `elements`, the elements of a non-null list with the list nullability
    /// `nullability`.
    pub(super) fn try_new(
        elements: ArrayRef,
        nullability: Nullability,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let is_empty = elements.is_empty();

        // A null element never equals a needle, so the probe is better off without it.
        let valid = elements.validity()?.execute_mask(elements.len(), ctx)?;
        let has_null_element = !valid.all_true();
        let probe_elements = if has_null_element {
            elements.filter(valid)?
        } else {
            elements.clone()
        };

        let probe = new_probe(probe_elements, ctx)?;

        Ok(Self {
            elements,
            nullability,
            set: Arc::new(ElementSet {
                probe,
                has_null_element,
                is_empty,
            }),
        })
    }

    /// The elements of the list that every row holds.
    pub fn elements(&self) -> &ArrayRef {
        &self.elements
    }

    /// The dtype of the list that every row holds.
    fn list_dtype(&self) -> DType {
        DType::List(Arc::new(self.elements.dtype().clone()), self.nullability)
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
        let (bits, needle_validity) = self.set.probe.contains(needles, ctx)?;
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
        let (bits, needle_validity) = self.set.probe.contains(&needles, ctx)?;
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

        let nullability = options.result_nullability(&self.list_dtype(), needle_dtype);

        let validity = if self.set.is_empty && !options.sql_null_semantics {
            Validity::NonNullable
        } else if options.sql_null_semantics && self.set.has_null_element {
            // Only a match is known. A comparison with the null element makes a non-match unknown.
            needle_validity.and(Validity::from(bits.clone()))?
        } else {
            needle_validity
        };

        Ok(BoolArray::new(bits, validity.union_nullability(nullability)).into_array())
    }

    /// Fails when needles of `needle_dtype` cannot be elements of the list.
    fn check_needle_dtype(&self, needle_dtype: &DType) -> VortexResult<()> {
        let element_dtype = self.elements.dtype();
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
            .field("elements", &self.elements)
            .field("nullability", &self.nullability)
            .finish_non_exhaustive()
    }
}

impl Display for PreparedSetData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "elements: {}", self.elements.len())
    }
}

impl ArrayHash for PreparedSetData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.elements.array_hash(state, accuracy);
        self.nullability.hash(state);
    }
}

impl ArrayEq for PreparedSetData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.nullability == other.nullability && self.elements.array_eq(&other.elements, accuracy)
    }
}

impl Array<PreparedSet> {
    /// Prepares `elements`, the elements of a non-null list with the list nullability
    /// `nullability`, as a set repeated `len` times.
    ///
    /// [`ListContains`] prepares a constant list itself. A kernel crate can use this to test its
    /// [`ListContainsElementKernel`] against a prepared set.
    ///
    /// [`ListContains`]: crate::scalar_fn::fns::list_contains::ListContains
    /// [`ListContainsElementKernel`]: crate::scalar_fn::fns::list_contains::ListContainsElementKernel
    ///
    /// # Errors
    ///
    /// Fails when the elements cannot be executed into the probe.
    pub fn try_new(
        elements: ArrayRef,
        nullability: Nullability,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let data = PreparedSetData::try_new(elements, nullability, ctx)?;
        Ok(Self::from_data(data, len))
    }

    /// An array of `len` rows that share the prepared set `data`.
    fn from_data(data: PreparedSetData, len: usize) -> Self {
        let dtype = data.list_dtype();

        // SAFETY: the dtype is the dtype of the list that every row holds.
        unsafe { Array::from_parts_unchecked(ArrayParts::new(PreparedSet, dtype, len, data)) }
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
            &data.list_dtype() == dtype,
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
        let data = array.data();
        let n_elements = data.elements.len();

        // Every row has the same offset and size, so use the narrowest width that fits the list.
        let (offsets, sizes) = match_smallest_list_offset_type!(n_elements, |O| {
            let size =
                O::try_from(n_elements).vortex_expect("list length fits the chosen offset type");
            (
                ConstantArray::new::<O>(O::default(), array.len()).into_array(),
                ConstantArray::new::<O>(size, array.len()).into_array(),
            )
        });

        // SAFETY: every view points at the range [0, n_elements) of the elements, and the list
        // is never null.
        let list = unsafe {
            ListViewArray::new_unchecked(
                data.elements.clone(),
                offsets,
                sizes,
                Validity::from(data.nullability),
            )
        };

        Ok(ExecutionResult::done(list))
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
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let elements = (0..array.elements.len())
            .map(|idx| array.elements.execute_scalar(idx, ctx))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Scalar::list(
            array.elements.dtype().clone(),
            elements,
            array.nullability,
        ))
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
            PreparedSetArray::from_data(array.data().clone(), range.len()).into_array(),
        ))
    }
}

impl FilterReduce for PreparedSet {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            PreparedSetArray::from_data(array.data().clone(), mask.true_count()).into_array(),
        ))
    }
}
