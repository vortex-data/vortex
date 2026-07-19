use std::collections::HashMap;
use std::str::FromStr;

use http::{HeaderMap, HeaderName, HeaderValue};
use object_store::{ClientConfigKey, ClientOptions};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{PyDict, PyString};

use crate::config::PyConfigValue;
use crate::error::PyObjectStoreError;

/// A wrapper around `ClientConfigKey` that implements [`FromPyObject`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyClientConfigKey(ClientConfigKey);

impl<'py> FromPyObject<'_, 'py> for PyClientConfigKey {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let s = obj.extract::<PyBackedStr>()?.to_lowercase();
        let key = s.parse().map_err(PyObjectStoreError::ObjectStoreError)?;
        Ok(Self(key))
    }
}

impl<'py> IntoPyObject<'py> for PyClientConfigKey {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self.0.as_ref()))
    }
}

impl<'py> IntoPyObject<'py> for &PyClientConfigKey {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self.0.as_ref()))
    }
}

/// A wrapper around `ClientOptions` that implements [`FromPyObject`].
#[derive(Clone, Debug, PartialEq)]
pub struct PyClientOptions {
    string_options: HashMap<PyClientConfigKey, PyConfigValue>,
    default_headers: Option<PyHeaderMap>,
}

impl<'py> FromPyObject<'_, 'py> for PyClientOptions {
    type Error = PyErr;

    // Need custom extraction because default headers is not a string value
    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let dict = obj.extract::<Bound<PyDict>>()?;
        let mut string_options = HashMap::new();
        let mut default_headers = None;

        for (key, value) in dict.iter() {
            if let Ok(key) = key.extract::<PyClientConfigKey>() {
                string_options.insert(key, value.extract::<PyConfigValue>()?);
            } else {
                let key = key.extract::<PyBackedStr>()?;
                if &key == "default_headers" {
                    default_headers = Some(value.extract::<PyHeaderMap>()?);
                } else {
                    return Err(PyValueError::new_err(format!("Invalid key: {key}.")));
                }
            }
        }

        Ok(Self {
            string_options,
            default_headers,
        })
    }
}

impl<'py> IntoPyObject<'py> for PyClientOptions {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = self.string_options.into_pyobject(py)?;
        if let Some(headers) = self.default_headers {
            dict.set_item("default_headers", headers)?;
        }
        Ok(dict)
    }
}

impl<'py> IntoPyObject<'py> for &PyClientOptions {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = (&self.string_options).into_pyobject(py)?;
        if let Some(headers) = &self.default_headers {
            dict.set_item("default_headers", headers)?;
        }
        Ok(dict.clone())
    }
}

impl From<PyClientOptions> for ClientOptions {
    fn from(value: PyClientOptions) -> Self {
        let mut options = ClientOptions::new();
        for (key, value) in value.string_options.into_iter() {
            options = options.with_config(key.0, value.0);
        }

        if let Some(headers) = value.default_headers {
            options = options.with_default_headers(headers.0);
        }

        options
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PyHeaderMap(HeaderMap);

impl<'py> FromPyObject<'_, 'py> for PyHeaderMap {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let dict = obj.extract::<Bound<PyDict>>()?;
        let mut header_map = HeaderMap::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let key = HeaderName::from_str(&key.extract::<PyBackedStr>()?)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;

            // HTTP Header values can have non-ascii bytes, so first try to extract as bytes.
            let value = if let Ok(value_bytes) = value.extract::<PyBackedBytes>() {
                HeaderValue::from_bytes(&value_bytes)
            } else {
                HeaderValue::from_str(&value.extract::<PyBackedStr>()?)
            }
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

            header_map.insert(key, value);
        }
        Ok(Self(header_map))
    }
}

impl<'py> IntoPyObject<'py> for PyHeaderMap {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        for (key, value) in self.0.iter() {
            dict.set_item(key.as_str(), value.as_bytes())?;
        }
        Ok(dict)
    }
}

impl<'py> IntoPyObject<'py> for &PyHeaderMap {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        for (key, value) in self.0.iter() {
            dict.set_item(key.as_str(), value.as_bytes())?;
        }
        Ok(dict)
    }
}
