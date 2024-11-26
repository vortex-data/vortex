use std::ops::Deref;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::{vortex_bail, vortex_panic, VortexError, VortexResult};

use crate::value::ScalarValue;
use crate::{InnerScalarValue, Scalar};

pub struct ListScalar<'a> {
    dtype: &'a DType,
    elements: Option<Arc<[InnerScalarValue]>>,
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
                value: ScalarValue(value.clone()),
            })
    }

    pub fn elements(&self) -> impl Iterator<Item = Scalar> + '_ {
        self.elements
            .as_ref()
            .map(AsRef::as_ref)
            .unwrap_or_else(|| &[] as &[InnerScalarValue])
            .iter()
            .map(|e| Scalar {
                dtype: self.element_dtype(),
                value: ScalarValue(e.clone()),
            })
    }

    pub fn cast(&self, _dtype: &DType) -> VortexResult<Scalar> {
        todo!()
    }
}

impl Scalar {
    pub fn list(element_dtype: Arc<DType>, children: Vec<Scalar>) -> Self {
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
            dtype: DType::List(element_dtype, NonNullable),
            value: ScalarValue(InnerScalarValue::List(
                children
                    .into_iter()
                    .map(|x| x.value.0)
                    .collect::<Arc<[_]>>(),
            )),
        }
    }
}

impl<'a> TryFrom<&'a Scalar> for ListScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        if !matches!(value.dtype(), DType::List(..)) {
            vortex_bail!("Expected list scalar, found {}", value.dtype())
        }

        Ok(Self {
            dtype: value.dtype(),
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
