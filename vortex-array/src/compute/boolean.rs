use vortex_error::{vortex_bail, VortexResult};

use crate::array::BoolArray;
use crate::{ArrayDType, ArrayData, IntoArrayVariant};
pub trait AndFn {
    /// Point-wise logical _and_ between two Boolean arrays.
    ///
    /// This method uses Arrow-style null propagation rather than the Kleene logic semantics.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_array::ArrayData;
    /// use vortex_array::compute::and;
    /// use vortex_array::IntoCanonical;
    /// use vortex_array::accessor::ArrayAccessor;
    /// let a = ArrayData::from(vec![Some(true), Some(true), Some(true), None, None, None, Some(false), Some(false), Some(false)]);
    /// let b = ArrayData::from(vec![Some(true), None, Some(false), Some(true), None, Some(false), Some(true), None, Some(false)]);
    /// let result = and(a, b)?.into_canonical()?.into_bool()?;
    /// let result_vec = result.with_iterator(|it| it.map(|x| x.cloned()).collect::<Vec<_>>())?;
    /// assert_eq!(result_vec, vec![Some(true), None, Some(false), None, None, None, Some(false), None, Some(false)]);
    /// # use vortex_error::VortexError;
    /// # Ok::<(), VortexError>(())
    /// ```
    fn and(&self, array: &ArrayData) -> VortexResult<ArrayData>;

    /// Point-wise Kleene logical _and_ between two Boolean arrays.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_array::ArrayData;
    /// use vortex_array::compute::and_kleene;
    /// use vortex_array::IntoCanonical;
    /// use vortex_array::accessor::ArrayAccessor;
    /// let a = ArrayData::from(vec![Some(true), Some(true), Some(true), None, None, None, Some(false), Some(false), Some(false)]);
    /// let b = ArrayData::from(vec![Some(true), None, Some(false), Some(true), None, Some(false), Some(true), None, Some(false)]);
    /// let result = and_kleene(a, b)?.into_canonical()?.into_bool()?;
    /// let result_vec = result.with_iterator(|it| it.map(|x| x.cloned()).collect::<Vec<_>>())?;
    /// assert_eq!(result_vec, vec![Some(true), None, Some(false), None, None, Some(false), Some(false), Some(false), Some(false)]);
    /// # use vortex_error::VortexError;
    /// # Ok::<(), VortexError>(())
    /// ```
    fn and_kleene(&self, array: &ArrayData) -> VortexResult<ArrayData>;
}

pub trait OrFn {
    /// Point-wise logical _or_ between two Boolean arrays.
    ///
    /// This method uses Arrow-style null propagation rather than the Kleene logic semantics.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_array::ArrayData;
    /// use vortex_array::compute::or;
    /// use vortex_array::IntoCanonical;
    /// use vortex_array::accessor::ArrayAccessor;
    /// let a = ArrayData::from(vec![Some(true), Some(true), Some(true), None, None, None, Some(false), Some(false), Some(false)]);
    /// let b = ArrayData::from(vec![Some(true), None, Some(false), Some(true), None, Some(false), Some(true), None, Some(false)]);
    /// let result = or(a, b)?.into_canonical()?.into_bool()?;
    /// let result_vec = result.with_iterator(|it| it.map(|x| x.cloned()).collect::<Vec<_>>())?;
    /// assert_eq!(result_vec, vec![Some(true), None, Some(true), None, None, None, Some(true), None, Some(false)]);
    /// # use vortex_error::VortexError;
    /// # Ok::<(), VortexError>(())
    /// ```
    fn or(&self, array: &ArrayData) -> VortexResult<ArrayData>;

    /// Point-wise Kleene logical _or_ between two Boolean arrays.
    ///
    /// # Examples
    ///
    /// ```
    /// use vortex_array::ArrayData;
    /// use vortex_array::compute::or_kleene;
    /// use vortex_array::IntoCanonical;
    /// use vortex_array::accessor::ArrayAccessor;
    /// let a = ArrayData::from(vec![Some(true), Some(true), Some(true), None, None, None, Some(false), Some(false), Some(false)]);
    /// let b = ArrayData::from(vec![Some(true), None, Some(false), Some(true), None, Some(false), Some(true), None, Some(false)]);
    /// let result = or_kleene(a, b)?.into_canonical()?.into_bool()?;
    /// let result_vec = result.with_iterator(|it| it.map(|x| x.cloned()).collect::<Vec<_>>())?;
    /// assert_eq!(result_vec, vec![Some(true), Some(true), Some(true), Some(true), None, None, Some(true), None, Some(false)]);
    /// # use vortex_error::VortexError;
    /// # Ok::<(), VortexError>(())
    /// ```
    fn or_kleene(&self, array: &ArrayData) -> VortexResult<ArrayData>;
}

fn lift_boolean_operator<F, G>(
    lhs: impl AsRef<ArrayData>,
    rhs: impl AsRef<ArrayData>,
    trait_fun: F,
    bool_array_fun: G,
) -> VortexResult<ArrayData>
where
    F: Fn(&ArrayData, &ArrayData) -> Option<VortexResult<ArrayData>>,
    G: FnOnce(BoolArray, &ArrayData) -> VortexResult<ArrayData>,
{
    let lhs = lhs.as_ref();
    let rhs = rhs.as_ref();

    if lhs.len() != rhs.len() {
        vortex_bail!("Boolean operations aren't supported on arrays of different lengths")
    }

    if !lhs.dtype().is_boolean() || !rhs.dtype().is_boolean() {
        vortex_bail!("Boolean operations are only supported on boolean arrays")
    }

    if let Some(selection) = trait_fun(lhs, rhs) {
        return selection;
    }

    if let Some(selection) = trait_fun(rhs, lhs) {
        return selection;
    }

    // If neither side implements the trait, we try to expand the left-hand side into a `BoolArray`,
    // which we know does implement it, and call into that implementation.
    let lhs = lhs.clone().into_bool()?;

    bool_array_fun(lhs, rhs)
}

pub fn and(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    lift_boolean_operator(
        lhs,
        rhs,
        |lhs, rhs| lhs.with_dyn(|lhs| lhs.and().map(|lhs| lhs.and(rhs))),
        |lhs, rhs| lhs.and(rhs),
    )
}

pub fn and_kleene(
    lhs: impl AsRef<ArrayData>,
    rhs: impl AsRef<ArrayData>,
) -> VortexResult<ArrayData> {
    lift_boolean_operator(
        lhs,
        rhs,
        |lhs, rhs| lhs.with_dyn(|lhs| lhs.and_kleene().map(|lhs| lhs.and_kleene(rhs))),
        |lhs, rhs| lhs.and_kleene(rhs),
    )
}

pub fn or(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    lift_boolean_operator(
        lhs,
        rhs,
        |lhs, rhs| lhs.with_dyn(|lhs| lhs.or().map(|lhs| lhs.or(rhs))),
        |lhs, rhs| lhs.or(rhs),
    )
}

pub fn or_kleene(
    lhs: impl AsRef<ArrayData>,
    rhs: impl AsRef<ArrayData>,
) -> VortexResult<ArrayData> {
    lift_boolean_operator(
        lhs,
        rhs,
        |lhs, rhs| lhs.with_dyn(|lhs| lhs.or_kleene().map(|lhs| lhs.or_kleene(rhs))),
        |lhs, rhs| lhs.or_kleene(rhs),
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::array::BoolArray;
    use crate::compute::unary::scalar_at;
    use crate::IntoArrayData;

    #[rstest]
    #[case(BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter())
    .into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter())
    .into_array())]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter()).into_array())]
    fn test_or(#[case] lhs: ArrayData, #[case] rhs: ArrayData) {
        let r = or(&lhs, &rhs).unwrap();

        let r = r.into_bool().unwrap().into_array();

        let v0 = scalar_at(&r, 0).unwrap().value().as_bool().unwrap();
        let v1 = scalar_at(&r, 1).unwrap().value().as_bool().unwrap();
        let v2 = scalar_at(&r, 2).unwrap().value().as_bool().unwrap();
        let v3 = scalar_at(&r, 3).unwrap().value().as_bool().unwrap();

        assert!(v0.unwrap());
        assert!(v1.unwrap());
        assert!(v2.unwrap());
        assert!(!v3.unwrap());
    }

    #[rstest]
    #[case(BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter())
    .into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter())
    .into_array())]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter()).into_array())]
    fn test_and(#[case] lhs: ArrayData, #[case] rhs: ArrayData) {
        let r = and(&lhs, &rhs).unwrap().into_bool().unwrap().into_array();

        let v0 = scalar_at(&r, 0).unwrap().value().as_bool().unwrap();
        let v1 = scalar_at(&r, 1).unwrap().value().as_bool().unwrap();
        let v2 = scalar_at(&r, 2).unwrap().value().as_bool().unwrap();
        let v3 = scalar_at(&r, 3).unwrap().value().as_bool().unwrap();

        assert!(v0.unwrap());
        assert!(!v1.unwrap());
        assert!(!v2.unwrap());
        assert!(!v3.unwrap());
    }
}
