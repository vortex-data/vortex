use std::sync::Arc;

use object_store::aws::AmazonS3Builder;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreScheme};
use url::Url;
use vortex::error::{vortex_bail, VortexResult};
use vortex::io::ObjectStoreReadAt;

fn better_parse_url(url_str: &str) -> VortexResult<(Box<dyn ObjectStore>, Path)> {
    let url = Url::parse(url_str)?;

    let (scheme, path) = ObjectStoreScheme::parse(&url).map_err(object_store::Error::from)?;
    let store: Box<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => Box::new(LocalFileSystem::default()),
        ObjectStoreScheme::AmazonS3 => {
            Box::new(AmazonS3Builder::from_env().with_url(url_str).build()?)
        }
        ObjectStoreScheme::GoogleCloudStorage => Box::new(
            GoogleCloudStorageBuilder::from_env()
                .with_url(url_str)
                .build()?,
        ),
        ObjectStoreScheme::MicrosoftAzure => Box::new(
            MicrosoftAzureBuilder::from_env()
                .with_url(url_str)
                .build()?,
        ),
        ObjectStoreScheme::Http => Box::new(
            HttpBuilder::new()
                .with_url(&url[..url::Position::BeforePath])
                .build()?,
        ),
        otherwise => vortex_bail!("unrecognized object store scheme: {:?}", otherwise),
    };

    Ok((store, path))
}

pub async fn vortex_read_at_from_url(url: &str) -> VortexResult<ObjectStoreReadAt> {
    let (object_store, location) = better_parse_url(url)?;
    Ok(ObjectStoreReadAt::new(Arc::from(object_store), location))
}
