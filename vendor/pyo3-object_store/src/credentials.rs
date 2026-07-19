use chrono::Utc;
use chrono::{DateTime, TimeDelta};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::future::Future;
use tokio::sync::Mutex;

/// A temporary authentication token with an associated expiry
#[derive(Debug, Clone)]
pub(crate) struct TemporaryToken<T> {
    /// The temporary credential
    pub token: T,
    /// The instant at which this credential is no longer valid
    /// None means the credential does not expire
    pub expiry: Option<DateTime<Utc>>,
}

/// Provides [`TokenCache::get_or_insert_with`] which can be used to cache a
/// [`TemporaryToken`] based on its expiry
#[derive(Debug)]
pub(crate) struct TokenCache<T> {
    /// A temporary token and the instant at which it was fetched
    cache: Mutex<Option<(TemporaryToken<T>, DateTime<Utc>)>>,
    min_ttl: TimeDelta,
    /// How long to wait before re-attempting a token fetch after receiving one that
    /// is still within the min-ttl
    fetch_backoff: TimeDelta,
}

impl<T> Default for TokenCache<T> {
    fn default() -> Self {
        Self {
            cache: Default::default(),
            min_ttl: TimeDelta::seconds(300),
            fetch_backoff: TimeDelta::milliseconds(100),
        }
    }
}

impl<T: Clone> Clone for TokenCache<T> {
    /// Cloning the token cache invalidates the cache.
    fn clone(&self) -> Self {
        Self {
            cache: Default::default(),
            min_ttl: self.min_ttl,
            fetch_backoff: self.fetch_backoff,
        }
    }
}

impl<T: Clone + Send> TokenCache<T> {
    /// Override the minimum remaining TTL for a cached token to be used
    pub(crate) fn with_min_ttl(self, min_ttl: TimeDelta) -> Self {
        Self { min_ttl, ..self }
    }

    pub(crate) async fn get_or_insert_with<F, Fut, E>(&self, f: F) -> Result<T, E>
    where
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<TemporaryToken<T>, E>> + Send,
    {
        // let now = Instant::now();
        let now = Utc::now();

        let mut locked = self.cache.lock().await;

        if let Some((cached, fetched_at)) = locked.as_ref() {
            match cached.expiry {
                Some(expiry_time) => {
                    // let x = ttl - now;
                    // let x = ttl.signed_duration_since(now);
                    // let x = expiry_time - now > self.min_ttl.into();
                    if expiry_time - now > self.min_ttl ||
                        // if we've recently attempted to fetch this token and it's not actually
                        // expired, we'll wait to re-fetch it and return the cached one
                        (Utc::now() - fetched_at < self.fetch_backoff && expiry_time - now > TimeDelta::zero())
                    {
                        return Ok(cached.token.clone());
                    }
                }
                None => return Ok(cached.token.clone()),
            }
        }

        let cached = f().await?;
        let token = cached.token.clone();
        *locked = Some((cached, Utc::now()));

        Ok(token)
    }
}

/// Check whether a Python object is awaitable
pub(crate) fn is_awaitable(ob: &Bound<PyAny>) -> PyResult<bool> {
    let py = ob.py();
    let inspect_mod = py.import(intern!(py, "inspect"))?;
    inspect_mod
        .call_method1(intern!(py, "isawaitable"), PyTuple::new(py, [ob])?)?
        .extract::<bool>()
}
