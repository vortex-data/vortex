// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::arrays::ConstantArray;
use crate::arrays::List;
use crate::arrays::Primitive;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

/// The elements data array containing all list elements concatenated together.
pub(super) const ELEMENTS_SLOT: usize = 0;
/// The offsets array defining the start/end of each list within the elements array.
pub(super) const OFFSETS_SLOT: usize = 1;
/// The validity bitmap indicating which list elements are non-null.
pub(super) const VALIDITY_SLOT: usize = 2;
pub(super) const NUM_SLOTS: usize = 3;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["elements", "offsets", "validity"];

/// A list array that stores variable-length lists of elements, similar to `Vec<Vec<T>>`.
///
/// This mirrors the Apache Arrow List array encoding and provides efficient storage
/// for nested data where each row contains a list of elements of the same type.
///
/// ## Data Layout
///
/// The list array uses an offset-based encoding:
/// - **Elements array**: A flat array containing all list elements concatenated together
/// - **Offsets array**: Integer array where `offsets[i]` is an (inclusive) start index into
///   the **elements** and `offsets[i+1]` is the (exclusive) stop index for the `i`th list.
/// - **Validity**: Optional mask indicating which lists are null
///
/// This allows for excellent cascading compression of the elements and offsets, as similar values
/// are clustered together and the offsets have a predictable pattern and small deltas between
/// consecutive elements.
///
/// ## Offset Semantics
///
/// - Offsets must be non-nullable integers (i32, i64, etc.)
/// - Offsets array has length `n+1` where `n` is the number of lists
/// - List `i` contains elements from `elements[offsets[i]..offsets[i+1]]`
/// - Offsets must be monotonically increasing
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::{ListArray, PrimitiveArray};
/// use vortex_array::arrays::list::ListArrayExt;
/// use vortex_array::validity::Validity;
/// use vortex_array::IntoArray;
/// use vortex_buffer::buffer;
/// use std::sync::Arc;
///
/// // Create a list array representing [[1, 2], [3, 4, 5], []]
/// let elements = buffer![1i32, 2, 3, 4, 5].into_array();
/// let offsets = buffer![0u32, 2, 5, 5].into_array(); // 3 lists
///
/// let list_array = ListArray::try_new(
///     elements.into_array(),
///     offsets.into_array(),
///     Validity::NonNullable,
/// ).unwrap();
///
/// assert_eq!(list_array.len(), 3);
///
/// // Access individual lists
/// let first_list = list_array.list_elements_at(0).unwrap();
/// assert_eq!(first_list.len(), 2); // [1, 2]
///
/// let third_list = list_array.list_elements_at(2).unwrap();
/// assert!(third_list.is_empty()); // []
/// ```
#[derive(Clone, Debug, Default)]
pub struct ListData;

impl Display for ListData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub struct ListDataParts {
    pub elements: ArrayRef,
    pub offsets: ArrayRef,
    pub validity: Validity,
    pub dtype: DType,
}

impl ListData {
    pub(crate) fn make_slots(
        elements: &ArrayRef,
        offsets: &ArrayRef,
        validity: &Validity,
        len: usize,
    ) -> Vec<Option<ArrayRef>> {
        vec![
            Some(elements.clone()),
            Some(offsets.clone()),
            validity_to_child(validity, len),
        ]
    }

    /// Creates a new `ListArray`.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in `ListArray::new_unchecked`.
    pub fn build(elements: ArrayRef, offsets: ArrayRef, validity: Validity) -> Self {
        Self::try_build(elements, offsets, validity).vortex_expect("ListArray new")
    }

    /// Constructs a new `ListArray`.
    ///
    /// See `ListArray::new_unchecked` for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// `ListArray::new_unchecked`.
    pub(crate) fn try_build(
        elements: ArrayRef,
        offsets: ArrayRef,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::validate(&elements, &offsets, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked() })
    }

    /// Creates a new `ListArray` without validation from these components:
    ///
    /// * `elements` is a flat array containing all list elements concatenated.
    /// * `offsets` is an integer array where `offsets[i]` is the start index for list `i`.
    /// * `validity` holds the null values.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// - Offsets must be a non-nullable integer array.
    /// - Offsets must have at least one element (even for empty lists, it should contain \[0\]).
    /// - Offsets must be sorted (monotonically increasing).
    /// - All offset values must be non-negative.
    /// - The maximum offset must not exceed `elements.len()`.
    /// - If validity is an array, its length must equal `offsets.len() - 1`.
    pub unsafe fn new_unchecked() -> Self {
        Self
    }

    /// Validates the components that would be used to create a `ListArray`.
    ///
    /// This function checks all the invariants required by `ListArray::new_unchecked`.
    pub fn validate(
        elements: &ArrayRef,
        offsets: &ArrayRef,
        validity: &Validity,
    ) -> VortexResult<()> {
        // Offsets must have at least one element
        vortex_ensure!(
            !offsets.is_empty(),
            InvalidArgument: "Offsets must have at least one element, [0] for an empty list"
        );

        // Offsets must be of integer type, and cannot go lower than 0.
        vortex_ensure!(
            offsets.dtype().is_int() && !offsets.dtype().is_nullable(),
            InvalidArgument: "offsets have invalid type {}",
            offsets.dtype()
        );

        // We can safely unwrap the DType as primitive now
        let offsets_ptype = offsets.dtype().as_ptype();

        // Offsets must be sorted (but not strictly sorted, zero-length lists are allowed)
        if let Some(is_sorted) = offsets.statistics().compute_is_sorted() {
            vortex_ensure!(is_sorted, InvalidArgument: "offsets must be sorted");
        } else {
            vortex_bail!(InvalidArgument: "offsets must report is_sorted statistic");
        }

        // Validate that offsets min is non-negative, and max does not exceed the length of
        // the elements array.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        if let Some(min_max) = min_max(offsets, &mut ctx)? {
            match_each_integer_ptype!(offsets_ptype, |P| {
                #[allow(clippy::absurd_extreme_comparisons, unused_comparisons)]
                {
                    let max = min_max
                        .max
                        .as_primitive()
                        .as_::<P>()
                        .vortex_expect("offsets type must fit offsets values");
                    let min = min_max
                        .min
                        .as_primitive()
                        .as_::<P>()
                        .vortex_expect("offsets type must fit offsets values");

                    vortex_ensure!(
                        min >= 0,
                        InvalidArgument: "offsets minimum {min} outside valid range [0, {max}]"
                    );

                    vortex_ensure!(
                        max <= P::try_from(elements.len()).unwrap_or_else(|_| vortex_panic!(
                            "Offsets type {} must be able to fit elements length {}",
                            <P as NativePType>::PTYPE,
                            elements.len()
                        )),
                        InvalidArgument: "Max offset {max} is beyond the length of the elements array {}",
                        elements.len()
                    );
                }
            })
        } else {
            // TODO(aduffy): fallback to slower validation pathway?
            vortex_bail!(
                InvalidArgument: "offsets array with encoding {} must support min_max compute function",
                offsets.encoding_id()
            );
        };

        // If a validity array is present, it must be the same length as the ListArray
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == offsets.len() - 1,
                InvalidArgument: "validity with size {validity_len} does not match array size {}",
                offsets.len() - 1
            );
        }

        Ok(())
    }
    // TODO(connor)[ListView]: Create 2 functions `reset_offsets` and `recursive_reset_offsets`,
    // where `reset_offsets` is infallible.
    // Also, `reset_offsets` can be made more efficient by replacing `sub_scalar` with a match on
    // the offset type and manual subtraction and fast path where `offsets[0] == 0`.
}

pub trait ListArrayExt: TypedArrayRef<List> {
    fn nullability(&self) -> crate::dtype::Nullability {
        match self.as_ref().dtype() {
            DType::List(_, nullability) => *nullability,
            _ => unreachable!("ListArrayExt requires a list dtype"),
        }
    }

    fn elements(&self) -> &ArrayRef {
        self.as_ref().slots()[ELEMENTS_SLOT]
            .as_ref()
            .vortex_expect("ListArray elements slot")
    }

    fn offsets(&self) -> &ArrayRef {
        self.as_ref().slots()[OFFSETS_SLOT]
            .as_ref()
            .vortex_expect("ListArray offsets slot")
    }

    fn list_validity(&self) -> Validity {
        child_to_validity(&self.as_ref().slots()[VALIDITY_SLOT], self.nullability())
    }

    fn list_validity_mask(&self) -> vortex_mask::Mask {
        self.list_validity().to_mask(self.as_ref().len())
    }

    fn offset_at(&self, index: usize) -> VortexResult<usize> {
        vortex_ensure!(
            index <= self.as_ref().len(),
            "Index {index} out of bounds 0..={}",
            self.as_ref().len()
        );

        if let Some(p) = self.offsets().as_opt::<Primitive>() {
            Ok(match_each_native_ptype!(p.ptype(), |P| {
                p.as_slice::<P>()[index].as_()
            }))
        } else {
            self.offsets()
                .scalar_at(index)?
                .as_primitive()
                .as_::<usize>()
                .ok_or_else(|| vortex_error::vortex_err!("offset value does not fit in usize"))
        }
    }

    fn list_elements_at(&self, index: usize) -> VortexResult<ArrayRef> {
        let start = self.offset_at(index)?;
        let end = self.offset_at(index + 1)?;
        self.elements().slice(start..end)
    }

    fn sliced_elements(&self) -> VortexResult<ArrayRef> {
        let start = self.offset_at(0)?;
        let end = self.offset_at(self.as_ref().len())?;
        self.elements().slice(start..end)
    }

    fn element_dtype(&self) -> &DType {
        self.elements().dtype()
    }

    fn reset_offsets(&self, recurse: bool) -> VortexResult<Array<List>> {
        let mut elements = self.sliced_elements()?;
        if recurse && elements.is_canonical() {
            elements = elements.to_canonical()?.compact()?.into_array();
        } else if recurse && let Some(child_list_array) = elements.as_opt::<List>() {
            elements = child_list_array
                .into_owned()
                .reset_offsets(recurse)?
                .into_array();
        }

        let offsets = self.offsets();
        let first_offset = offsets.scalar_at(0)?;
        let adjusted_offsets = offsets.clone().binary(
            ConstantArray::new(first_offset, offsets.len()).into_array(),
            Operator::Sub,
        )?;

        Array::<List>::try_new(elements, adjusted_offsets, self.list_validity())
    }
}
impl<T: TypedArrayRef<List>> ListArrayExt for T {}

impl Array<List> {
    /// Creates a new `ListArray`.
    pub fn new(elements: ArrayRef, offsets: ArrayRef, validity: Validity) -> Self {
        let dtype = DType::List(Arc::new(elements.dtype().clone()), validity.nullability());
        let len = offsets.len().saturating_sub(1);
        let slots = ListData::make_slots(&elements, &offsets, &validity, len);
        let data = ListData::build(elements, offsets, validity);
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(List, dtype, len, data).with_slots(slots))
        }
    }

    /// Constructs a new `ListArray`.
    pub fn try_new(
        elements: ArrayRef,
        offsets: ArrayRef,
        validity: Validity,
    ) -> VortexResult<Self> {
        let dtype = DType::List(Arc::new(elements.dtype().clone()), validity.nullability());
        let len = offsets.len().saturating_sub(1);
        let slots = ListData::make_slots(&elements, &offsets, &validity, len);
        let data = ListData::try_build(elements, offsets, validity)?;
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(List, dtype, len, data).with_slots(slots))
        })
    }

    /// Creates a new `ListArray` without validation.
    ///
    /// # Safety
    ///
    /// See [`ListData::new_unchecked`].
    pub unsafe fn new_unchecked(elements: ArrayRef, offsets: ArrayRef, validity: Validity) -> Self {
        let dtype = DType::List(Arc::new(elements.dtype().clone()), validity.nullability());
        let len = offsets.len().saturating_sub(1);
        let slots = ListData::make_slots(&elements, &offsets, &validity, len);
        let data = unsafe { ListData::new_unchecked() };
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(List, dtype, len, data).with_slots(slots))
        }
    }

    pub fn into_data_parts(self) -> ListDataParts {
        let dtype = self.dtype().clone();
        let elements = self.slots()[ELEMENTS_SLOT]
            .clone()
            .vortex_expect("ListArray elements slot");
        let offsets = self.slots()[OFFSETS_SLOT]
            .clone()
            .vortex_expect("ListArray offsets slot");
        let validity = self.list_validity();
        ListDataParts {
            elements,
            offsets,
            validity,
            dtype,
        }
    }
}
