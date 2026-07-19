use std::sync::Arc;

use object_store::memory::InMemory;
use object_store::ObjectStoreScheme;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use pyo3::{intern, IntoPyObjectExt};

use crate::error::GenericError;
use crate::retry::PyRetryConfig;
use crate::url::PyUrl;
use crate::{
    PyAzureStore, PyClientOptions, PyGCSStore, PyHttpStore, PyLocalStore, PyMemoryStore,
    PyObjectStoreResult, PyS3Store,
};

/// Simple construction of stores by url.
// Note: We don't extract the PyObject in the function signature because it's possible that
// AWS/Azure/Google config keys could overlap. And so we don't want to accidentally parse a config
// as an AWS config before knowing that the URL scheme is AWS.
#[pyfunction]
#[pyo3(signature = (url, *, config=None, client_options=None, retry_config=None, credential_provider=None, **kwargs))]
pub fn from_url<'py>(
    py: Python<'py>,
    url: PyUrl,
    config: Option<Bound<'py, PyAny>>,
    client_options: Option<PyClientOptions>,
    retry_config: Option<PyRetryConfig>,
    credential_provider: Option<Bound<'py, PyAny>>,
    kwargs: Option<Bound<'py, PyAny>>,
) -> PyObjectStoreResult<Bound<'py, PyAny>> {
    let (scheme, _) = ObjectStoreScheme::parse(url.as_ref()).map_err(object_store::Error::from)?;
    match scheme {
        ObjectStoreScheme::AmazonS3 => PyS3Store::from_url(
            &PyType::new::<PyS3Store>(py),
            url,
            config.map(|x| x.extract()).transpose()?,
            client_options,
            retry_config,
            credential_provider.map(|x| x.extract()).transpose()?,
            kwargs.map(|x| x.extract()).transpose()?,
        ),
        ObjectStoreScheme::GoogleCloudStorage => PyGCSStore::from_url(
            &PyType::new::<PyGCSStore>(py),
            url,
            config.map(|x| x.extract()).transpose()?,
            client_options,
            retry_config,
            credential_provider.map(|x| x.extract()).transpose()?,
            kwargs.map(|x| x.extract()).transpose()?,
        ),
        ObjectStoreScheme::MicrosoftAzure => PyAzureStore::from_url(
            &PyType::new::<PyAzureStore>(py),
            url,
            config.map(|x| x.extract()).transpose()?,
            client_options,
            retry_config,
            credential_provider.map(|x| x.extract()).transpose()?,
            kwargs.map(|x| x.extract()).transpose()?,
        ),
        ObjectStoreScheme::Http => {
            raise_if_config_passed(config, kwargs, "http")?;
            PyHttpStore::from_url(
                &PyType::new::<PyHttpStore>(py),
                py,
                url,
                client_options,
                retry_config,
            )
        }
        ObjectStoreScheme::Local => {
            let mut automatic_cleanup = false;
            let mut mkdir = false;
            if let Some(kwargs) = kwargs {
                let kwargs = kwargs.cast::<PyDict>()?;
                if let Some(val) = kwargs.get_item(intern!(py, "automatic_cleanup"))? {
                    automatic_cleanup = val.extract()?;
                }
                if let Some(val) = kwargs.get_item(intern!(py, "mkdir"))? {
                    mkdir = val.extract()?;
                }
            }

            PyLocalStore::from_url(
                &PyType::new::<PyLocalStore>(py),
                url,
                automatic_cleanup,
                mkdir,
            )
        }
        ObjectStoreScheme::Memory => {
            raise_if_config_passed(config, kwargs, "memory")?;
            let store: PyMemoryStore = Arc::new(InMemory::new()).into();
            Ok(store.into_bound_py_any(py)?)
        }
        scheme => Err(GenericError::new_err(format!("Unknown URL scheme {scheme:?}")).into()),
    }
}

fn raise_if_config_passed(
    config: Option<Bound<PyAny>>,
    kwargs: Option<Bound<PyAny>>,
    scheme: &str,
) -> PyObjectStoreResult<()> {
    if config.is_some() || kwargs.is_some() {
        return Err(GenericError::new_err(format!(
            "Cannot pass config or keyword parameters for scheme {scheme:?}"
        ))
        .into());
    }
    Ok(())
}
