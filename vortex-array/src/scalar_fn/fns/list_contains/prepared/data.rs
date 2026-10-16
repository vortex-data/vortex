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
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use super::PreparedSet;
use crate::AnyCanonical;
use crate::ArrayHaystack;
use crate::ArrayRef;
use crate::Canonical;
use crate::CanonicalView;
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
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::literal::Literal;
use crate::validity::Validity;

/// The options of a [`PreparedSet`]: the list that every row holds, and its set.
///
/// The set is shared by every clone, so every batch a [`PreparedSet`] is applied to, and every
/// slice or filter of the array it gives, looks up needles in the one set. The set looks up
/// needles through an [`ArrayHaystack`]: one of your own, given to [`Self::try_with_haystack`],
/// or else one that [`ArrayHaystack::try_new`] builds on the first lookup, since it needs an
/// [`ExecutionCtx`] to materialize the elements.
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
    /// The haystack of the non-null elements, given or built on the first lookup.
    haystack: OnceLock<SharedVortexResult<ArrayHaystack>>,
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
            haystack: OnceLock::new(),
        })))
    }

    /// Prepares the non-null list scalar `list` as a set that looks up needles through
    /// `haystack`, a haystack of your own over the elements of the list.
    ///
    /// # Errors
    ///
    /// Fails when `list` is not a list, or is null, or when the lookup array of `haystack` does
    /// not have the element dtype of the list.
    pub fn try_with_haystack(list: Scalar, haystack: ArrayHaystack) -> VortexResult<Self> {
        let set = Self::try_new(list)?;
        vortex_ensure!(
            set.element_dtype()
                .eq_ignore_nullability(haystack.array().dtype()),
            "A haystack of {} cannot look up elements of {}",
            haystack.array().dtype(),
            set.element_dtype()
        );

        if set.0.haystack.set(Ok(haystack)).is_err() {
            vortex_panic!("A new prepared set has no haystack yet");
        }
        Ok(set)
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

    /// The haystack that looks up needles among the elements, built on the first call unless
    /// one was given, and shared after it.
    ///
    /// A kernel of a needle encoding can read the lookup array of the haystack, for example to
    /// look up its own values in a lookup array of its own encoding.
    ///
    /// # Errors
    ///
    /// Fails when the elements cannot be executed into a haystack.
    pub fn haystack(&self, ctx: &mut ExecutionCtx) -> VortexResult<&ArrayHaystack> {
        self.0
            .haystack
            .get_or_init(|| {
                // The canonical form of a constant list holds its elements once, whatever its
                // length.
                ConstantArray::new(self.list().clone(), 1)
                    .into_array()
                    .execute::<ListViewArray>(ctx)
                    .and_then(|list| ArrayHaystack::try_new(list.elements(), ctx))
                    .map_err(Arc::new)
            })
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
    /// A constant needle is looked up once, and gives a constant result. Canonical needles are
    /// looked up in the haystack. Any other needles are not decoded here: the result is a lazy
    /// [`ListContains`] over them, so that their encoding chooses how to look them up, through a
    /// [`ListContainsElementKernel`] at any step of its decoding. A kernel therefore passes the
    /// parts of its array, such as the values of a dictionary, and never the array itself.
    ///
    /// [`ListContainsElementKernel`]: crate::scalar_fn::fns::list_contains::ListContainsElementKernel
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
        match needles.as_opt::<AnyCanonical>() {
            Some(canonical) => self.lookup(canonical, needles.dtype(), options, ctx),
            None => Ok(ListContains::try_new_opts(
                PreparedSet::new_array(self.clone(), needles.len()),
                needles.clone(),
                *options,
            )?
            .into_array()),
        }
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
        let needles = ConstantArray::new(needle.clone(), 1)
            .into_array()
            .execute::<Canonical>(ctx)?
            .into_array();
        let needles = needles
            .as_opt::<AnyCanonical>()
            .vortex_expect("A canonical array matches AnyCanonical");
        self.lookup(needles, needle.dtype(), options, ctx)?
            .execute_scalar(0, ctx)
    }

    /// Looks up canonical needles in the haystack.
    fn lookup(
        &self,
        needles: CanonicalView<'_>,
        needle_dtype: &DType,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let (bits, needle_validity) = self.haystack(ctx)?.contains(needles, ctx)?;
        self.result_from_bits(bits, needle_validity, needle_dtype, options)
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
