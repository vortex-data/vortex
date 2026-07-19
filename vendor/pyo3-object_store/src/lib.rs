#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod api;
mod aws;
mod azure;
mod client;
mod config;
mod credentials;
pub(crate) mod error;
mod gcp;
mod http;
mod local;
mod memory;
mod path;
mod prefix;
mod retry;
mod simple;
mod store;
mod url;

pub use api::{register_exceptions_module, register_store_module};
pub use aws::PyS3Store;
pub use azure::PyAzureStore;
pub use client::{PyClientConfigKey, PyClientOptions};
pub use error::{PyObjectStoreError, PyObjectStoreResult};
pub use gcp::PyGCSStore;
pub use http::PyHttpStore;
pub use local::PyLocalStore;
pub use memory::PyMemoryStore;
pub use path::PyPath;
pub use prefix::MaybePrefixedStore;
pub use simple::from_url;
pub use store::{AnyObjectStore, PyExternalObjectStore, PyObjectStore};
pub use url::PyUrl;
