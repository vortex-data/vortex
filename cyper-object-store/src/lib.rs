//! An `ObjectStore` that runs IO operations on top of [compio].
//!
//! `CyperS3` supports S3-compatible object storage services, and uses
//! an embedded compio runtime to implement object store requests.

mod object_store;
mod signer;

use std::fmt::Display;
use std::io;
use std::sync::Arc;
use std::time::SystemTime;

use ::object_store::aws::AwsCredentialProvider;
use ::object_store::path::Path;
use ::object_store::GetRange;
use cyper::{Client, Response};
use http::header::RANGE;
use signer::sign_request;
use url::Url;

/// An `ObjectStore` implementation for S3-compatible services, where requests
/// execute in io_uring via the [compio] async runtime.
///
/// `CyperS3` implements all methods for reading to S3-compatible object storage,
/// and can be used anywhere an `ObjectStore` or `Arc<dyn ObjectStore>` is expected.
///
/// The futures returned by `CyperS3` can only be executed within a compio context.
#[derive(Debug, Clone)]
pub struct CyperS3 {
    client: Client,
    base_url: Url,
    config: S3Config,
}

impl Display for CyperS3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CyperS3").finish_non_exhaustive()
    }
}

/// S3 configuration parameters.
#[derive(Debug, Clone)]
pub struct S3Config {
    pub bucket: Arc<str>,
    pub region: Arc<str>,
    pub endpoint: Option<Arc<str>>,
    pub credentials: AwsCredentialProvider,
    pub virtual_host: bool,
}

/// Infer a default endpoint URL from a bucket + region combo.
fn region_endpoint(bucket: &str, region: &str) -> io::Result<Url> {
    let url = if region == "us-east-1" {
        format!("https://{bucket}.s3.amazonaws.com")
    } else {
        format!("https://{bucket}.{region}.s3.amazonaws.com")
    };
    Url::parse(url.as_str())
        .map_err(|parse_error| io::Error::new(io::ErrorKind::Other, parse_error))
}

impl CyperS3 {
    /// Create a new S3 `ObjectStore` using a [cyper] client.
    pub fn new(config: S3Config) -> io::Result<Self> {
        let base_url = if let Some(ref endpoint) = config.endpoint {
            Url::parse(endpoint.as_ref())
                .map_err(|parse_error| io::Error::new(io::ErrorKind::Other, parse_error))?
        } else {
            region_endpoint(config.bucket.as_ref(), config.region.as_ref())?
        };

        let client = cyper::ClientBuilder::new().build();

        Ok(Self {
            client,
            base_url,
            config,
        })
    }
}

// Create a new custom ObjectStoreError here instead.

impl CyperS3 {
    /// Get a set of bytes for the given object at the optional range.
    ///
    /// We want to return a GetResult instead here...
    pub async fn get_byte_range(
        &self,
        path: &Path,
        range: Option<&GetRange>,
    ) -> io::Result<Response> {
        let creds = self.config.credentials.get_credential().await?;

        // In virtual-hosting style, the hostname should start with the bucket name.
        // If we are not in virtual-hosting style, we prefix HTTP request paths with the bucket name.
        let url = if self.config.virtual_host {
            self.base_url
                .join(path.as_ref())
                .map_err(|parse_error| io::Error::new(io::ErrorKind::Other, parse_error))?
        } else {
            self.base_url
                .join(format!("{}/{}", self.config.bucket.as_ref(), path.as_ref()).as_ref())
                .map_err(|parse_error| io::Error::new(io::ErrorKind::Other, parse_error))?
        };

        let mut request = self
            .client
            .get(url)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        if let Some(ref range) = range {
            request = request
                .header(RANGE, format!("{range}"))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        }
        let request = request.build();
        let signed = sign_request(
            request,
            SystemTime::now(),
            creds.key_id.as_str(),
            creds.secret_key.as_str(),
            self.config.region.as_ref(),
        );

        self.client
            .execute(signed)
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }
}
