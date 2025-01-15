use std::ops::Deref;
use std::sync::Arc;

use itertools::Itertools as _;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, vortex_panic, VortexError, VortexExpect as _, VortexResult};

use crate::value::{InnerScalarValue, ScalarValue};
use crate::Scalar;

pub struct ListScalar<'a> {
    dtype: &'a DType,
    element_dtype: &'a Arc<DType>,
    elements: Option<Arc<[ScalarValue]>>,
}

impl<'a> ListScalar<'a> {
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.elements.as_ref().map(|e| e.len()).unwrap_or(0)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self.elements.as_ref() {
            None => true,
            Some(l) => l.is_empty(),
        }
    }

    #[inline]
    pub fn is_null(&self) -> bool {
        self.elements.is_none()
    }

    pub fn element_dtype(&self) -> DType {
        let DType::List(element_type, _) = self.dtype() else {
            unreachable!();
        };
        (*element_type).deref().clone()
    }

    pub fn element(&self, idx: usize) -> Option<Scalar> {
        self.elements
            .as_ref()
            .and_then(|l| l.get(idx))
            .map(|value| Scalar {
                dtype: self.element_dtype(),
                value: value.clone(),
            })
    }

    pub fn elements(&self) -> impl Iterator<Item = Scalar> + '_ {
        self.elements
            .as_ref()
            .map(AsRef::as_ref)
            .unwrap_or_else(|| &[] as &[ScalarValue])
            .iter()
            .map(|e| Scalar {
                dtype: self.element_dtype(),
                value: e.clone(),
            })
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        let DType::List(element_dtype, ..) = dtype else {
            vortex_bail!("Can't cast {:?} to {}", self.dtype(), dtype)
        };

        Ok(Scalar::new(
            dtype.clone(),
            ScalarValue(InnerScalarValue::List(
                self.elements
                    .as_ref()
                    .vortex_expect("nullness handled in Scalar::cast")
                    .iter()
                    .map(|element| {
                        Scalar::new(DType::clone(self.element_dtype), element.clone())
                            .cast(element_dtype)
                            .map(|x| x.value().clone())
                    })
                    .process_results(|iter| iter.collect())?,
            )),
        ))
    }
}

impl Scalar {
    pub fn list(
        element_dtype: Arc<DType>,
        children: Vec<Scalar>,
        nullability: Nullability,
    ) -> Self {
        for child in &children {
            if child.dtype() != &*element_dtype {
                vortex_panic!(
                    "tried to create list of {} with values of type {}",
                    element_dtype,
                    child.dtype()
                );
            }
        }
        Self {
            dtype: DType::List(element_dtype, nullability),
            value: ScalarValue(InnerScalarValue::List(
                children.into_iter().map(|x| x.value).collect::<Arc<[_]>>(),
            )),
        }
    }

    pub fn list_empty(element_dtype: Arc<DType>, nullability: Nullability) -> Self {
        Self {
            dtype: DType::List(element_dtype, nullability),
            value: ScalarValue(InnerScalarValue::Null),
        }
    }
}

impl<'a> TryFrom<&'a Scalar> for ListScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        let DType::List(element_dtype, ..) = value.dtype() else {
            vortex_bail!("Expected list scalar, found {}", value.dtype())
        };

        Ok(Self {
            dtype: value.dtype(),
            element_dtype,
            elements: value.value.as_list()?.cloned(),
        })
    }
}

impl<'a, T: for<'b> TryFrom<&'b Scalar, Error = VortexError>> TryFrom<&'a Scalar> for Vec<T> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        let value = ListScalar::try_from(value)?;
        let mut elems = vec![];
        for e in value.elements() {
            elems.push(T::try_from(&e)?);
        }
        Ok(elems)
    }
}
