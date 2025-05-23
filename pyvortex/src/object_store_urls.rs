use std::sync::Arc;

use object_store::aws::AmazonS3Builder;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreScheme};
use url::Url;
use vortex::error::{VortexResult, vortex_bail};

pub(crate) fn object_store_from_url(
    url_str: &str,
) -> VortexResult<(ObjectStoreScheme, Arc<dyn ObjectStore>, Path)> {
    let url = Url::parse(url_str)?;

    let (scheme, path) = ObjectStoreScheme::parse(&url).map_err(object_store::Error::from)?;
    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => Arc::new(LocalFileSystem::default()),
        ObjectStoreScheme::AmazonS3 => {
            Arc::new(AmazonS3Builder::from_env().with_url(url_str).build()?)
        }
        ObjectStoreScheme::GoogleCloudStorage => Arc::new(
            GoogleCloudStorageBuilder::from_env()
                .with_url(url_str)
                .build()?,
        ),
        ObjectStoreScheme::MicrosoftAzure => Arc::new(
            MicrosoftAzureBuilder::from_env()
                .with_url(url_str)
                .build()?,
        ),
        ObjectStoreScheme::Http => Arc::new(
            HttpBuilder::new()
                .with_url(&url[..url::Position::BeforePath])
                .build()?,
        ),
        otherwise => vortex_bail!("unrecognized object store scheme: {:?}", otherwise),
    };

    Ok((scheme, store, path))
}
