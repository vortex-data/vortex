#![allow(clippy::tests_outside_test_module, clippy::panic_in_result_fn)]

use std::io;
use std::sync::Arc;

use bytes::Bytes;
use cyper_object_store::{CyperS3, S3Config};
use local_s3::LocalS3;
use object_store::aws::AwsCredential;
use object_store::path::Path;
use object_store::{ObjectStore, StaticCredentialProvider};

const BUCKET: &str = "test-bucket";
const KEY_ID: &str = "key1";
const SECRET_KEY: &str = "secret1";

/// Integration test against a mock S3 service adapter.
#[compio::test]
async fn test_e2e() -> io::Result<()> {
    let tempdir = tempfile::tempdir()?;
    let path = tempdir.path().to_path_buf();

    let local_s3 = LocalS3::new(path).with_credentials(KEY_ID, SECRET_KEY);
    local_s3.create_bucket(BUCKET);
    local_s3.put_object(
        BUCKET,
        "object1",
        Bytes::from_static("0123456789".as_bytes()),
    );

    // Launch the S3 service.
    local_s3.start().detach();

    let static_creds = Arc::new(StaticCredentialProvider::new(AwsCredential {
        key_id: KEY_ID.to_string(),
        secret_key: SECRET_KEY.to_string(),
        token: None,
    }));

    // Make the object store work instead.
    let object_store: Arc<dyn ObjectStore> = Arc::new(CyperS3::new(S3Config {
        bucket: BUCKET.into(),
        credentials: static_creds,
        region: "us-east-1".into(),
        endpoint: Some("http://localhost:3030".into()),
        virtual_host: false,
    })?);

    let get_range = object_store.get_range(&Path::from("object1"), 0..4).await?;
    assert_eq!(get_range.as_ref(), "0123".as_bytes());

    let get_all = object_store
        .get(&Path::from("object1"))
        .await?
        .bytes()
        .await?;
    assert_eq!(get_all.as_ref(), "0123456789".as_bytes());

    Ok(())
}
