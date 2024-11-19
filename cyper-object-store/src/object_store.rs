use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use cyper::Response;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use http::header::{
    CACHE_CONTROL, CONTENT_DISPOSITION, CONTENT_ENCODING, CONTENT_LANGUAGE, CONTENT_LENGTH,
    CONTENT_RANGE, CONTENT_TYPE, ETAG, LAST_MODIFIED,
};
use http::{HeaderMap, StatusCode};
use object_store::path::Path;
use object_store::{
    Attribute, Attributes, Error as ObjectStoreError, GetOptions, GetRange, GetResult,
    GetResultPayload, ListResult, MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOpts,
    PutOptions, PutPayload, PutResult,
};

use crate::CyperS3;

const STORE: &str = "CYPER_S3";

macro_rules! make_object_store_error {
    ($err:expr) => {
        ::object_store::Error::Generic {
            store: STORE,
            source: $err.into(),
        }
    };
}

#[async_trait]
impl ObjectStore for CyperS3 {
    /// Save the provided `payload` to `location` with the given options
    async fn put_opts(
        &self,
        _location: &Path,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> Result<PutResult, ObjectStoreError> {
        Err(ObjectStoreError::NotSupported {
            source: "operation not supported for CyperS3 object_store".into(),
        })
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _options: PutMultipartOpts,
    ) -> Result<Box<dyn MultipartUpload>, ObjectStoreError> {
        Err(ObjectStoreError::NotSupported {
            source: "operation not supported for CyperS3 object_store".into(),
        })
    }

    /// Implement the GET operation, optionally with a range header.
    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> Result<GetResult, ObjectStoreError> {
        let response = self
            .get_byte_range(location, options.range.as_ref())
            .await
            .map_err(|io_error| make_object_store_error!(io_error))?;

        match response.status() {
            StatusCode::NOT_FOUND => Err(ObjectStoreError::NotFound {
                path: location.to_string(),
                source: "403 forbidden".into(),
            }),
            _ => get_result(location, options.range, response),
        }
    }

    async fn delete(&self, _location: &Path) -> Result<(), ObjectStoreError> {
        Err(ObjectStoreError::NotSupported {
            source: "operation not supported for CyperS3 object_store".into(),
        })
    }

    fn list(&self, _prefix: Option<&Path>) -> BoxStream<'_, Result<ObjectMeta, ObjectStoreError>> {
        futures::stream::once(async move {
            Err(ObjectStoreError::NotSupported {
                source: "operation not supported for CyperS3 object_store".into(),
            })
        })
        .boxed()
    }

    async fn list_with_delimiter(
        &self,
        _prefix: Option<&Path>,
    ) -> Result<ListResult, ObjectStoreError> {
        Err(ObjectStoreError::NotSupported {
            source: "operation not supported for CyperS3 object_store".into(),
        })
    }

    async fn copy(&self, _from: &Path, _to: &Path) -> Result<(), ObjectStoreError> {
        Err(ObjectStoreError::NotSupported {
            source: "operation not supported for CyperS3 object_store".into(),
        })
    }

    async fn copy_if_not_exists(&self, _from: &Path, _to: &Path) -> Result<(), ObjectStoreError> {
        Err(ObjectStoreError::NotSupported {
            source: "operation not supported for CyperS3 object_store".into(),
        })
    }
}

fn get_result(
    location: &Path,
    range: Option<GetRange>,
    response: Response,
) -> Result<GetResult, ObjectStoreError> {
    let mut meta = header_meta(location, response.headers())?;

    // ensure that we receive the range we asked for
    let range = if range.is_some() {
        let val = response.headers().get(CONTENT_RANGE).ok_or_else(|| {
            make_object_store_error!("Content-Range header required if range request made")
        })?;

        let value = val.to_str().map_err(|err| make_object_store_error!(err))?;
        let value = ContentRange::from_str(value)
            .ok_or_else(|| make_object_store_error!("Invalid value for Content-Range header"))?;
        let actual = value.range;

        // Update size to reflect full size of object (#5272)
        meta.size = value.size;

        actual
    } else {
        0..meta.size
    };

    macro_rules! parse_attributes {
        ($headers:expr, $(($header:expr, $attr:expr, $err:expr)),*) => {{
            let mut attributes = Attributes::new();
            $(
            if let Some(x) = $headers.get($header) {
                let x = x.to_str().map_err(|_| make_object_store_error!($err))?;
                attributes.insert($attr, x.to_string().into());
            }
            )*
            attributes
        }}
    }

    let mut attributes = parse_attributes!(
        response.headers(),
        (
            CACHE_CONTROL,
            Attribute::CacheControl,
            "Cache-Control header value invalid"
        ),
        (
            CONTENT_DISPOSITION,
            Attribute::ContentDisposition,
            "Content-Disposition header value invalid"
        ),
        (
            CONTENT_ENCODING,
            Attribute::ContentEncoding,
            "Content-Encoding header value invalid"
        ),
        (
            CONTENT_LANGUAGE,
            Attribute::ContentLanguage,
            "Content-Languge header value invalid"
        ),
        (
            CONTENT_TYPE,
            Attribute::ContentType,
            "Content-Type header value invalid"
        )
    );

    // Add attributes that match the user-defined metadata prefix (e.g. x-amz-meta-)
    for (key, val) in response.headers() {
        if let Some(suffix) = key
            .as_str()
            .strip_prefix(USER_DEFINED_METADATA_HEADER_PREFIX)
        {
            if let Ok(val_str) = val.to_str() {
                attributes.insert(
                    Attribute::Metadata(suffix.to_string().into()),
                    val_str.to_string().into(),
                );
            } else {
                return Err(ObjectStoreError::Generic {
                    store: STORE,
                    source: format!("invalid metadata header {}", key.as_str()).into(),
                });
            }
        }
    }

    let body_fut = response
        .bytes()
        .map(|res| res.map_err(|err| make_object_store_error!(err)))
        .boxed();

    Ok(GetResult {
        range,
        meta,
        attributes,
        payload: GetResultPayload::Stream(futures::stream::once(body_fut).boxed()),
    })
}

use core::ops::Range;

struct ContentRange {
    /// The range of the object returned
    range: Range<usize>,
    /// The total size of the object being requested
    size: usize,
}

impl ContentRange {
    /// Parse a content range of the form `bytes <range-start>-<range-end>/<size>`
    ///
    /// <https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Range>
    fn from_str(s: &str) -> Option<Self> {
        let rem = s.trim().strip_prefix("bytes ")?;
        let (range, size) = rem.split_once('/')?;
        let size = size.parse().ok()?;

        let (start_s, end_s) = range.split_once('-')?;

        let start = start_s.parse().ok()?;
        let end: usize = end_s.parse().ok()?;

        Some(Self {
            size,
            range: start..end + 1,
        })
    }
}

fn header_meta(location: &Path, headers: &HeaderMap) -> Result<ObjectMeta, ObjectStoreError> {
    let last_modified = match headers.get(LAST_MODIFIED) {
        Some(last_modified) => {
            let last_modified = last_modified
                .to_str()
                .map_err(|err| make_object_store_error!(err))?;
            DateTime::parse_from_rfc2822(last_modified)
                .map_err(|err| make_object_store_error!(err))?
                .with_timezone(&Utc)
        }
        None => Utc.timestamp_nanos(0),
    };

    let e_tag = match get_etag(headers) {
        Ok(e_tag) => e_tag,
        Err(e) => return Err(e),
    };

    let content_length = headers
        .get(CONTENT_LENGTH)
        .ok_or_else(|| make_object_store_error!("Content-Length header required"))?;

    let content_length = content_length
        .to_str()
        .map_err(|err| make_object_store_error!(err))?;
    let size = content_length
        .parse::<usize>()
        .map_err(|parse_err| ObjectStoreError::Generic {
            store: STORE,
            source: parse_err.into(),
        })?;

    let version = match headers.get(VERSION_HEADER) {
        Some(v) => Some(
            v.to_str()
                .map_err(|err| make_object_store_error!(err))?
                .to_string(),
        ),
        None => None,
    };

    Ok(ObjectMeta {
        location: location.clone(),
        last_modified,
        version,
        size,
        e_tag,
    })
}

fn get_etag(headers: &HeaderMap) -> Result<Option<String>, ObjectStoreError> {
    match headers.get(ETAG) {
        Some(etag) => etag
            .to_str()
            .map_err(|err| make_object_store_error!(err))
            .map(|etag| Some(etag.to_string())),
        None => Ok(None),
    }
}

const VERSION_HEADER: &str = "x-amz-version-id";
const USER_DEFINED_METADATA_HEADER_PREFIX: &str = "x-amz-meta-";
