//! Views into arrays of individual values.
//!
//! Vortex, like Arrow, avoids copying data. The classes in this package are returned by
//! :meth:`.Array.scalar_at`. They represent shared-memory views into individual values of a Vortex
//! array.

mod binary;
mod bool;
mod decimal;
mod extension;
pub mod factory;
mod into_py;
mod list;
mod null;
mod primitive;
mod struct_;
mod utf8;

use pyo3::PyClass;
use pyo3::prelude::*;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexExpect};
use vortex::scalar::Scalar;

use crate::dtype::PyDType;
use crate::scalar::binary::PyBinaryScalar;
use crate::scalar::bool::PyBoolScalar;
use crate::scalar::decimal::PyDecimalScalar;
use crate::scalar::extension::PyExtensionScalar;
use crate::scalar::list::PyListScalar;
use crate::scalar::null::PyNullScalar;
use crate::scalar::primitive::PyPrimitiveScalar;
use crate::scalar::struct_::PyStructScalar;
use crate::scalar::utf8::PyUtf8Scalar;
use crate::{PyVortex, install_module};

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "scalar")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.scalar", &m)?;

    m.add_function(wrap_pyfunction!(factory::scalar, &m)?)?;

    m.add_class::<PyScalar>()?;

    m.add_class::<PyBinaryScalar>()?;
    m.add_class::<PyBoolScalar>()?;
    m.add_class::<PyExtensionScalar>()?;
    m.add_class::<PyListScalar>()?;
    m.add_class::<PyNullScalar>()?;
    m.add_class::<PyPrimitiveScalar>()?;
    m.add_class::<PyDecimalScalar>()?;
    m.add_class::<PyUtf8Scalar>()?;
    m.add_class::<PyStructScalar>()?;

    Ok(())
}

/// Base class for Vortex scalar types.
#[pyclass(name = "Scalar", module = "vortex", subclass, frozen, eq, hash)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyScalar(Scalar);

/// A marker trait indicating a PyO3 class is a subclass of a Vortex `Scalar`.
pub trait ScalarSubclass: PyClass<BaseType = PyScalar> {
    type Scalar<'a>;
}

/// A trait for extracting a typed and borrowed scalar from a [`Scalar`].
///
/// This is functionally the same as `AsRef` trait, except that the result is an owned type
/// with a lifetime, instead of a reference with a lifetime.
pub trait AsScalarRef<'a, T: 'a> {
    fn as_scalar_ref(&'a self) -> T;
}

/// Implement downcasting a `PyScalar` per the subclass in the marker trait.
impl<'a, T: ScalarSubclass> AsScalarRef<'a, <T as ScalarSubclass>::Scalar<'a>> for PyRef<'a, T>
where
    for<'b> <T as ScalarSubclass>::Scalar<'b>: TryFrom<&'b Scalar, Error = VortexError>,
{
    fn as_scalar_ref(&self) -> <T as ScalarSubclass>::Scalar<'_> {
        <<T as ScalarSubclass>::Scalar<'_>>::try_from(self.as_super().inner())
            .vortex_expect("Failed to downcast scalar")
    }
}

impl PyScalar {
    /// Initialize a [`PyScalar`] from a Vortex [`Scalar`], ensuring the correct subclass is
    /// returned.
    pub fn init(py: Python, scalar: Scalar) -> PyResult<Bound<PyScalar>> {
        // TODO(ngates): Bound::as_super would be great, but it's in newer PyO3.
        match scalar.dtype() {
            DType::Null => Self::with_subclass(py, scalar, PyNullScalar),
            DType::Bool(_) => Self::with_subclass(py, scalar, PyBoolScalar),
            DType::Primitive(..) => Self::with_subclass(py, scalar, PyPrimitiveScalar),
            DType::Decimal(..) => Self::with_subclass(py, scalar, PyDecimalScalar),
            DType::Utf8(..) => Self::with_subclass(py, scalar, PyUtf8Scalar),
            DType::Binary(..) => Self::with_subclass(py, scalar, PyBinaryScalar),
            DType::Struct(..) => Self::with_subclass(py, scalar, PyStructScalar),
            DType::List(..) => Self::with_subclass(py, scalar, PyListScalar),
            DType::Extension(..) => Self::with_subclass(py, scalar, PyExtensionScalar),
        }
    }

    /// Initialize a [`PyScalar`] from a Vortex [`Scalar`], with the given subclass.
    /// We keep this a private method to ensure we correctly match on the scalar DType in init.
    fn with_subclass<S: PyClass<BaseType = PyScalar>>(
        py: Python,
        scalar: Scalar,
        subclass: S,
    ) -> PyResult<Bound<PyScalar>> {
        Ok(Bound::new(
            py,
            PyClassInitializer::from(PyScalar(scalar)).add_subclass(subclass),
        )?
        .into_any()
        .downcast_into::<PyScalar>()?)
    }

    /// Return the inner [`Scalar`] value.
    pub fn inner(&self) -> &Scalar {
        &self.0
    }
}

/// Define the interface methods of a `PyScalar`. Note that all children should override these
/// methods and there's currently no good way to do this in PyO3.
#[pymethods]
impl PyScalar {
    /// Return the :class:`~vortex.DType` of the scalar.
    #[getter]
    pub fn dtype(self_: PyRef<'_, Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(self_.py(), self_.0.dtype().clone())
    }

    /// Return the scalar value as a Python object.
    pub fn as_py(&self, py: Python) -> PyResult<PyObject> {
        PyVortex(&self.0).into_pyobject(py).map(|v| v.into())
    }
}
