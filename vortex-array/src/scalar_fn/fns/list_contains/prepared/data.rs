// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex_buffer::BitBuffer;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayHaystack;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ListViewArray;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::literal::Literal;
use crate::validity::Validity;

/// The options of a [`PreparedSet`]: the list that every row holds, and its set.
///
/// The set is shared by every clone, so every batch a [`PreparedSet`] is applied to, and every
/// slice or filter of the array it gives, looks up needles in the one set. The haystack needs
/// an [`ExecutionCtx`] to materialize the elements, so it is built on the first lookup.
///
/// [`PreparedSet`]: crate::scalar_fn::fns::list_contains::PreparedSet
#[derive(Clone)]
pub struct PreparedSetData(Arc<SharedSet>);

struct SharedSet {
    /// The [`Literal`] of the non-null list that every row holds, shared with the expression it
    /// came from rather than cloned out of it.
    literal: ScalarFnRef,
    /// Whether the list holds a null element, which the haystack does not hold.
    has_null_element: bool,
    /// Whether the list holds no element at all, counting null elements.
    is_empty: bool,
    /// The elements, and the haystack of the non-null ones, built on the first lookup.
    set: OnceLock<SharedVortexResult<ElementSet>>,
}

/// The elements of the list, and their haystack.
struct ElementSet {
    /// All elements of the list, null elements included, with the element dtype of the list.
    elements: ArrayRef,
    haystack: ArrayHaystack,
}

impl ElementSet {
    fn try_new(list: &Scalar, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // The canonical form of a constant list holds its elements once, whatever its length.
        let elements = ConstantArray::new(list.clone(), 1)
            .into_array()
            .execute::<ListViewArray>(ctx)?
            .elements()
            .clone();
        let haystack = ArrayHaystack::try_new(&elements, ctx)?;

        Ok(Self { elements, haystack })
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
                vortex_bail!(InvalidArgument: "A prepared set needs a literal list");
            };
            vortex_ensure!(
                matches!(list.dtype(), DType::List(..)),
                MismatchedTypes: "A prepared set needs a list, got {}",
                list.dtype()
            );
            let Some(elements) = list.as_list().element_values() else {
                vortex_bail!(InvalidArgument: "A prepared set needs a non-null list");
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
    /// Fails when the elements cannot be executed into the haystack.
    pub fn elements(&self, ctx: &mut ExecutionCtx) -> VortexResult<&ArrayRef> {
        Ok(&self.element_set(ctx)?.elements)
    }

    /// The elements and their haystack, built on the first call and shared after it.
    fn element_set<'a>(&'a self, ctx: &mut ExecutionCtx) -> VortexResult<&'a ElementSet> {
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
            vortex_panic!(AssertionFailed: "A prepared set always holds a list");
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
    /// A constant needle is looked up once, and gives a constant result.
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
        let (bits, needle_validity) = self.element_set(ctx)?.haystack.contains(needles, ctx)?;
        self.result_from_bits(bits, needle_validity, needles.dtype(), options)
    }

    /// Whether the constant `needle` is an element of the list, under `options`.
    ///
    /// The answer is the one [`Self::contains`] gives for each row of a needle with this value, for
    /// example for the fill value of a sparse needle. The haystack looks up one row.
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
        let (bits, needle_validity) = self.element_set(ctx)?.haystack.contains(&needles, ctx)?;
        self.result_from_bits(bits, needle_validity, needle.dtype(), options)?
            .execute_scalar(0, ctx)
    }

    /// Makes the result of [`Self::contains`] from one membership bit per needle.
    ///
    /// A kernel that finds the matches of its needles without this haystack, for example from
    /// bounds of its own, uses this to apply the null semantics of [`Self::contains`]. A set bit
    /// in `bits` tells that the needle equals an element. `needle_validity` is the validity of the
    /// needles, and `needle_dtype` is their dtype. The bit of a null needle has no effect.
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
                InvalidArgument: "Needle validity has {len} rows, but there are {} membership bits",
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
                MismatchedTypes: "Element type {} of list does not match search value {}",
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
