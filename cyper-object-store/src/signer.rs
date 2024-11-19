#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::time::SystemTime;

use aws_sigv4::sign::v4::{calculate_signature, generate_signing_key};
use chrono::{DateTime, Utc};
use cyper::Request;
use http::header::{AUTHORIZATION, HOST};
use http::{HeaderName, HeaderValue};
use sha2::{Digest, Sha256};

/// AWS signature algorithm used for signing
pub const SIGNATURE_ALGORITHM: &str = "AWS4-HMAC-SHA256";

/// SHA-256 hash digest of an empty string.
static EMPTY_CHECKSUM: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Service used for S3-compatible storage that uses the AWSV4 signature algorithm.
pub const S3_SERVICE: &str = "s3";

const X_AMAZON_CONTENT_SHA256: &str = "x-amz-content-sha256";
const X_AMAZON_DATE: &str = "x-amz-date";

/// Sign request using the AWS v4 signature algorithm.
pub fn sign_request(
    mut request: Request,
    now: SystemTime,
    key_id: &str,
    secret_key: &str,
    region: &str,
) -> Request {
    // Set the Host header.
    let host = &request.url()[url::Position::BeforeHost..url::Position::AfterPort];
    let host = host.to_string();
    request
        .headers_mut()
        .insert(HOST, HeaderValue::from_str(host.as_str()).unwrap());

    // Set date header to current timestamp
    let timestamp: DateTime<Utc> = now.into();
    let timestamp = timestamp.format("%Y%m%dT%H%M%SZ").to_string();
    request.headers_mut().insert(
        X_AMAZON_CONTENT_SHA256,
        HeaderValue::from_str(EMPTY_CHECKSUM).unwrap(),
    );
    request.headers_mut().insert(
        X_AMAZON_DATE,
        HeaderValue::from_str(timestamp.as_str()).unwrap(),
    );

    let (creq, signed_headers) = canonical_request(&request);
    let string_to_sign = string_to_sign(creq.as_str(), now, region);

    let signing_key = generate_signing_key(secret_key, now, region, S3_SERVICE);
    let sig = calculate_signature(signing_key, string_to_sign.as_bytes());

    // Set the Authorization header to the provided value.
    let authorization =
        authorization_header(sig.as_str(), now, region, key_id, signed_headers.as_str());

    request.headers_mut().insert(
        AUTHORIZATION,
        HeaderValue::from_str(authorization.as_str()).unwrap(),
    );

    request
}

/// Generate an authorization header usable with an S3-compatible service that uses
/// the AWSV4 signature algorithm.
///
/// See: https://docs.aws.amazon.com/images/IAM/latest/UserGuide/images/sigV4-using-auth-header.png
fn authorization_header(
    signature: &str,
    date: SystemTime,
    region: &str,
    key_id: &str,
    signed_headers: &str,
) -> String {
    let mut header_value = String::new();

    // Signature algorithm
    header_value.push_str(SIGNATURE_ALGORITHM);
    header_value.push(' ');

    // Credential scope
    let date_time: DateTime<Utc> = date.into();
    header_value.push_str("Credential=");
    header_value.push_str(key_id);
    header_value.push('/');
    header_value.push_str(format!("{}", date_time.format("%Y%m%d")).as_str());
    header_value.push('/');
    header_value.push_str(region);
    header_value.push('/');
    header_value.push_str("s3");
    header_value.push('/');
    header_value.push_str("aws4_request");

    // Signed headers
    header_value.push_str(", SignedHeaders=");
    header_value.push_str(signed_headers);

    // Signature (HMAC of StringToSign)
    header_value.push_str(", Signature=");
    header_value.push_str(signature);

    header_value
}

/// Generate the canonical string representation for an HTTP request, expected
/// as the input to the AWS V4 signing algorithm.
pub fn canonical_request(request: &Request) -> (String, String) {
    let mut canonical_request = String::new();
    // Method
    canonical_request.push_str(request.method().as_str());
    canonical_request.push('\n');

    // Path
    canonical_request.push_str(request.url().path());
    canonical_request.push('\n');

    // Query string
    canonical_request.push_str(request.url().query().unwrap_or_default());
    canonical_request.push('\n');

    // Canonical headers
    let mut canonical_headers = BTreeMap::new();
    let mut signed_headers = BTreeSet::new();
    for (name, value) in request.headers() {
        if let Ok(value_str) = value.to_str() {
            signed_headers.insert(canonical_header_name(name));
            canonical_headers.insert(canonical_header_name(name), value_str.to_string());
        }
    }
    for (name, value) in canonical_headers {
        canonical_request.push_str(format!("{name}:{value}\n").as_str());
    }
    canonical_request.push('\n');

    // Signed Headers
    let signed_headers = signed_headers.iter().cloned().collect::<Vec<_>>().join(";");
    canonical_request.push_str(signed_headers.as_str());
    canonical_request.push('\n');

    // TODO(aduffy): handle non-empty payloads
    // Hash of payload bytes
    canonical_request.push_str(EMPTY_CHECKSUM);

    (canonical_request, signed_headers)
}

fn string_to_sign(canonical_request: &str, now: SystemTime, region: &str) -> String {
    let mut to_sign = String::new();

    to_sign.push_str("AWS4-HMAC-SHA256\n");

    let timestamp: DateTime<Utc> = now.into();
    to_sign.push_str(timestamp.format("%Y%m%dT%H%M%SZ").to_string().as_str());
    to_sign.push('\n');

    let scope = format!("{}/{region}/s3/aws4_request\n", timestamp.format("%Y%m%d"));
    to_sign.push_str(scope.as_str());

    let hash = Sha256::digest(canonical_request);
    to_sign.push_str(hex::encode(hash).as_str());

    to_sign
}

fn canonical_header_name(name: &HeaderName) -> String {
    name.as_str().to_lowercase()
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use cyper::Client;
    use http::header::{AUTHORIZATION, RANGE};
    use http::HeaderValue;

    use super::sign_request;

    #[compio::test]
    async fn test_signer() {
        let client = Client::new();
        let request = client
            .get("https://thisisjustfake.s3.amazonaws.com/ranges/ten")
            .unwrap()
            .header(RANGE, HeaderValue::from_str("bytes=2-4").unwrap())
            .unwrap()
            .build();

        let key_id = "key_id_goes_here";
        let secret_key = "icanttellyouitssecret";

        let signed_request = sign_request(
            request,
            SystemTime::UNIX_EPOCH,
            key_id,
            secret_key,
            "us-east-1",
        );
        assert_eq!(signed_request.headers().get(AUTHORIZATION).unwrap(), "AWS4-HMAC-SHA256 Credential=key_id_goes_here/19700101/us-east-1/s3/aws4_request, SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, Signature=0c94c5091be6cd3193f1753414a4dfd6487bd0fab5a99d8f7bcd10e9c51db04d");
    }
}
