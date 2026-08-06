// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Python-facing wrapper for the Hugging Face Hub object store.
//!
//! Reading an `hf://` URL needs no store at all — the URL registry resolves it, taking credentials
//! from `HF_TOKEN` or the saved login. This class exists for the two things a URL cannot say: a
//! token held in a variable rather than the environment, and a read that must stay anonymous even
//! though the environment offers credentials.

use std::str::FromStr;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vortex::cloud::hf::HfConfig;
use vortex::cloud::hf::HfRepoType;
use vortex::cloud::hf::make_hf_store;

/// How the `token` argument was passed.
///
/// Mirrors `huggingface_hub`'s own convention so that a caller can pass the same value through.
#[derive(Debug, Clone, FromPyObject)]
enum TokenArg {
    /// `True` asks for the saved login, `False` forces an anonymous read.
    Flag(bool),
    /// An explicit token.
    Value(String),
}

/// A Hugging Face Hub object store, rooted at one repository and revision.
///
/// Construct it with explicit configuration and pass it to
/// ``vortex.io.read_url(path, store=hf_store)``, where ``path`` is a path within the repository.
#[pyclass(name = "HfStore", module = "vortex._lib", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct HfStore {
    store: Arc<dyn object_store::ObjectStore>,
}

impl HfStore {
    /// Clone the underlying object store as an `Arc<dyn ObjectStore>`.
    pub fn to_arc(&self) -> Arc<dyn object_store::ObjectStore> {
        Arc::clone(&self.store)
    }
}

#[pymethods]
impl HfStore {
    #[new]
    #[pyo3(signature = (
        repo_id,
        *,
        repo_type = "dataset",
        revision = None,
        token = None,
        endpoint = None,
    ))]
    fn new(
        repo_id: String,
        repo_type: &str,
        revision: Option<String>,
        token: Option<TokenArg>,
        endpoint: Option<String>,
    ) -> PyResult<Self> {
        let repo_type =
            HfRepoType::from_str(repo_type).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut config = HfConfig::from_env(repo_type, repo_id, revision);
        match token {
            // The default and `token=True` both mean "whatever the environment offers", which
            // `from_env` has already resolved.
            None | Some(TokenArg::Flag(true)) => {}
            Some(TokenArg::Flag(false)) => config.token = None,
            Some(TokenArg::Value(token)) => config.token = Some(token),
        }
        if let Some(endpoint) = endpoint {
            config.endpoint = endpoint;
        }

        let store = make_hf_store(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { store })
    }
}

/// Register the Hugging Face store class on the `vortex._lib` module.
pub(crate) fn init(_py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_class::<HfStore>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex::cloud::hf::HfConfig;
    use vortex::cloud::hf::HfRepoType;
    use vortex::cloud::hf::make_hf_store;

    /// `token=False` must reach `make_hf_store` with no token at all, which is the whole reason this
    /// class exists alongside plain `hf://` URL resolution.
    #[test]
    fn anonymous_config_builds() {
        let mut config = HfConfig::from_env(HfRepoType::Dataset, "org/name", None);
        config.token = None;

        assert!(make_hf_store(config).is_ok());
    }
}
