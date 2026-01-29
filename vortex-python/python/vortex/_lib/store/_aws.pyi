# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 Development Seed

from collections.abc import Coroutine
from datetime import datetime
from typing import Any, Literal, NotRequired, Protocol, Self, TypeAlias, TypedDict, Unpack

from ._client import ClientConfig
from ._retry import RetryConfig

S3Regions: TypeAlias = Literal[
    "af-south-1",
    "ap-east-1",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ap-southeast-5",
    "ap-southeast-7",
    "ca-central-1",
    "ca-west-1",
    "eu-central-1",
    "eu-central-2",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "il-central-1",
    "me-central-1",
    "me-south-1",
    "mx-central-1",
    "sa-east-1",
    "us-east-1",
    "us-east-2",
    "us-gov-east-1",
    "us-gov-west-1",
    "us-west-1",
    "us-west-2",
]
"""AWS regions."""

S3ChecksumAlgorithm: TypeAlias = Literal["SHA256"]
"""S3 Checksum algorithms

From https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity.html#using-additional-checksums
"""

S3EncryptionAlgorithm: TypeAlias = Literal[
    "AES256",
    "aws:kms",
    "aws:kms:dsse",
    "sse-c",
]

class S3Config(TypedDict, total=False):
    """Configuration parameters for S3Store.

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from vortex.store import S3Config
        ```
    """

    access_key_id: str
    """AWS Access Key.

    **Environment variable**: `AWS_ACCESS_KEY_ID`.
    """
    bucket: str
    """Bucket name (required).

    **Environment variables**:

    - `AWS_BUCKET`
    - `AWS_BUCKET_NAME`
    """
    checksum_algorithm: S3ChecksumAlgorithm | str
    """
    Sets the [checksum algorithm] which has to be used for object integrity check during upload.

    [checksum algorithm]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity.html

    **Environment variable**: `AWS_CHECKSUM_ALGORITHM`.
    """
    conditional_put: str
    """Configure how to provide conditional put support

    Supported values:

    - `"etag"` (default): Supported for S3-compatible stores that support conditional
        put using the standard [HTTP precondition] headers `If-Match` and
        `If-None-Match`.

        [HTTP precondition]: https://datatracker.ietf.org/doc/html/rfc9110#name-preconditions

    - `"dynamo:<TABLE_NAME>"` or `"dynamo:<TABLE_NAME>:<TIMEOUT_MILLIS>"`: The name of a DynamoDB table to
      use for coordination.

        This will use the same region, credentials and endpoint as configured for S3.

    **Environment variable**: `AWS_CONDITIONAL_PUT`.
    """
    container_credentials_relative_uri: str
    """Set the container credentials relative URI

    <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-iam-roles.html>

    **Environment variable**: `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`.
    """
    copy_if_not_exists: Literal["multipart"] | str
    """Configure how to provide "copy if not exists".

    Supported values:

    - `"multipart"`:

        Native Amazon S3 supports copy if not exists through a multipart upload
        where the upload copies an existing object and is completed only if the
        new object does not already exist.

        !!! warning
            When using this mode, `copy_if_not_exists` does not copy tags
            or attributes from the source object.

        !!! warning
            When using this mode, `copy_if_not_exists` makes only a best
            effort attempt to clean up the multipart upload if the copy operation
            fails. Consider using a lifecycle rule to automatically clean up
            abandoned multipart uploads.

    - `"header:<HEADER_NAME>:<HEADER_VALUE>"`:

        Some S3-compatible stores, such as Cloudflare R2, support copy if not exists
        semantics through custom headers.

        If set, `copy_if_not_exists` will perform a normal copy operation with the
        provided header pair, and expect the store to fail with `412 Precondition
        Failed` if the destination file already exists.

        For example `header: cf-copy-destination-if-none-match: *`, would set
        the header `cf-copy-destination-if-none-match` to `*`.

    - `"header-with-status:<HEADER_NAME>:<HEADER_VALUE>:<STATUS>"`:

        The same as the header variant above but allows custom status code checking, for
        object stores that return values other than 412.

    - `"dynamo:<TABLE_NAME>"` or `"dynamo:<TABLE_NAME>:<TIMEOUT_MILLIS>"`:

        The name of a DynamoDB table to use for coordination.

        The default timeout is used if not specified. This will use the same region,
        credentials and endpoint as configured for S3.

    **Environment variable**: `AWS_COPY_IF_NOT_EXISTS`.
    """
    default_region: S3Regions | str
    """Default region.

    **Environment variable**: `AWS_DEFAULT_REGION`.
    """
    disable_tagging: bool
    """Disable tagging objects. This can be desirable if not supported by the backing store.

    **Environment variable**: `AWS_DISABLE_TAGGING`.
    """
    endpoint: str
    """The endpoint for communicating with AWS S3.

    Defaults to the [region endpoint].

    For example, this might be set to `"http://localhost:4566:` for testing against a
    localstack instance.

    The `endpoint` field should be consistent with `with_virtual_hosted_style_request`,
    i.e. if `virtual_hosted_style_request` is set to `True` then `endpoint` should have
    the bucket name included.

    By default, only HTTPS schemes are enabled. To connect to an HTTP endpoint, enable
    `allow_http` in the client options.

    [region endpoint]: https://docs.aws.amazon.com/general/latest/gr/s3.html

    **Environment variables**:

    - `AWS_ENDPOINT_URL`
    - `AWS_ENDPOINT`
    """
    imdsv1_fallback: bool
    """Fall back to ImdsV1.

    By default instance credentials will only be fetched over [IMDSv2], as AWS
    recommends against having IMDSv1 enabled on EC2 instances as it is vulnerable to
    [SSRF attack]

    However, certain deployment environments, such as those running old versions of
    kube2iam, may not support IMDSv2. This option will enable automatic fallback to
    using IMDSv1 if the token endpoint returns a 403 error indicating that IMDSv2 is not
    supported.

    This option has no effect if not using instance credentials.

    [IMDSv2]: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-instance-metadata-service.html
    [SSRF attack]: https://aws.amazon.com/blogs/security/defense-in-depth-open-firewalls-reverse-proxies-ssrf-vulnerabilities-ec2-instance-metadata-service/

    **Environment variable**: `AWS_IMDSV1_FALLBACK`.
    """
    metadata_endpoint: str
    """Set the [instance metadata endpoint], used primarily within AWS EC2.

    This defaults to the IPv4 endpoint: `http://169.254.169.254`. One can alternatively
    use the IPv6 endpoint `http://fd00:ec2::254`.

    **Environment variable**: `AWS_METADATA_ENDPOINT`.
    """
    region: S3Regions | str
    """The region, defaults to `us-east-1`

    **Environment variable**: `AWS_REGION`.
    """
    request_payer: bool
    """If `True`, enable operations on requester-pays buckets.

    <https://docs.aws.amazon.com/AmazonS3/latest/userguide/RequesterPaysBuckets.html>

    **Environment variable**: `AWS_REQUEST_PAYER`.
    """
    s3_express: bool
    """Enable Support for S3 Express One Zone.

    **Environment variable**: `AWS_S3_EXPRESS`.
    """
    secret_access_key: str
    """Secret Access Key.

    **Environment variable**: `AWS_SECRET_ACCESS_KEY`.
    """
    server_side_encryption: S3EncryptionAlgorithm | str
    """Type of encryption to use.

    If set, must be one of:

    - `"AES256"` (SSE-S3)
    - `"aws:kms"` (SSE-KMS)
    - `"aws:kms:dsse"` (DSSE-KMS)
    - `"sse-c"`

    **Environment variable**: `AWS_SERVER_SIDE_ENCRYPTION`.
    """
    session_token: str
    """Token to use for requests (passed to underlying provider).

    **Environment variables**:

    - `AWS_SESSION_TOKEN`
    - `AWS_TOKEN`
    """
    skip_signature: bool
    """If `True`, S3Store will not fetch credentials and will not sign requests.

    This can be useful when interacting with public S3 buckets that deny authorized requests.

    **Environment variable**: `AWS_SKIP_SIGNATURE`.
    """
    sse_bucket_key_enabled: bool
    """Set whether to enable bucket key for server side encryption.

    This overrides the bucket default setting for bucket keys.

    - When `False`, each object is encrypted with a unique data key.
    - When `True`, a single data key is used for the entire bucket,
      reducing overhead of encryption.

    **Environment variable**: `AWS_SSE_BUCKET_KEY_ENABLED`.
    """
    sse_customer_key_base64: str
    """
    The base64 encoded, 256-bit customer encryption key to use for server-side
    encryption. If set, the server side encryption config value must be `"sse-c"`.

    **Environment variable**: `AWS_SSE_CUSTOMER_KEY_BASE64`.
    """
    sse_kms_key_id: str
    """
    The KMS key ID to use for server-side encryption.

    If set, the server side encryption config value must be `"aws:kms"` or `"aws:kms:dsse"`.

    **Environment variable**: `AWS_SSE_KMS_KEY_ID`.
    """
    unsigned_payload: bool
    """Avoid computing payload checksum when calculating signature.

    See [unsigned payload option](https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html).

    - `False` (default): Signed payload option is used, where the checksum for the request body is computed
      and included when constructing a canonical request.
    - `True`: Unsigned payload option is used. `UNSIGNED-PAYLOAD` literal is included when constructing a
       canonical request,

    **Environment variable**: `AWS_UNSIGNED_PAYLOAD`.
    """
    virtual_hosted_style_request: bool
    """If virtual hosted style request has to be used.

    If `virtual_hosted_style_request` is:

    - `False` (default):  Path style request is used
    - `True`:  Virtual hosted style request is used

    If the `endpoint` is provided then it should be consistent with
    `virtual_hosted_style_request`. i.e. if `virtual_hosted_style_request` is set to
    `True` then `endpoint` should have bucket name included.

    **Environment variable**: `AWS_VIRTUAL_HOSTED_STYLE_REQUEST`.
    """

class S3Credential(TypedDict):
    """An S3 credential.

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from vortex.store import S3Credential
        ```
    """

    access_key_id: str
    """AWS access key ID."""

    secret_access_key: str
    """AWS secret access key"""

    token: NotRequired[str | None]
    """AWS token."""

    expires_at: datetime | None
    """Expiry datetime of credential. The datetime should have time zone set.

    If None, the credential will never expire.
    """

class S3CredentialProvider(Protocol):
    """A type hint for a synchronous or asynchronous callback to provide custom S3 credentials.

    This should be passed into the `credential_provider` parameter of `S3Store`.

    **Examples:**

    Return static credentials that don't expire:
    ```py
    def get_credentials() -> S3Credential:
        return {
            "access_key_id": "...",
            "secret_access_key": "...",
            "token": None,
            "expires_at": None,
        }
    ```

    Return static credentials that are valid for 5 minutes:
    ```py
    from datetime import datetime, timedelta, UTC

    async def get_credentials() -> S3Credential:
        return {
            "access_key_id": "...",
            "secret_access_key": "...",
            "token": None,
            "expires_at": datetime.now(UTC) + timedelta(minutes=5),
        }
    ```

    A class-based credential provider with state:

    ```py
    from __future__ import annotations

    from typing import TYPE_CHECKING

    import boto3
    import botocore.credentials

    if TYPE_CHECKING:
        from vortex.store import S3Credential


    class Boto3CredentialProvider:
        credentials: botocore.credentials.Credentials

        def __init__(self, session: boto3.session.Session) -> None:
            credentials = session.get_credentials()
            if credentials is None:
                raise ValueError("Received None from session.get_credentials")

            self.credentials = credentials

        def __call__(self) -> S3Credential:
            frozen_credentials = self.credentials.get_frozen_credentials()
            return {
                "access_key_id": frozen_credentials.access_key,
                "secret_access_key": frozen_credentials.secret_key,
                "token": frozen_credentials.token,
                "expires_at": None,
            }
    ```

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from vortex.store import S3CredentialProvider
        ```
    """

    def __call__(self) -> S3Credential | Coroutine[Any, Any, S3Credential]:
        """Return an `S3Credential`."""

class S3Store:
    """Interface to an Amazon S3 bucket.

    All constructors will check for environment variables. Refer to
    [`S3Config`][vortex.store.S3Config] for valid environment variables.

    **Examples**:

    **Using requester-pays buckets**:

    Pass `request_payer=True` as a keyword argument or have `AWS_REQUESTER_PAYS=True`
    set in the environment.

    **Anonymous requests**:

    Pass `skip_signature=True` as a keyword argument or have `AWS_SKIP_SIGNATURE=True`
    set in the environment.
    """

    def __init__(  # type: ignore[misc] # Overlap between argument names and ** TypedDict items: "bucket"
        self,
        bucket: str | None = None,
        *,
        prefix: str | None = None,
        config: S3Config | None = None,
        client_options: ClientConfig | None = None,
        retry_config: RetryConfig | None = None,
        credential_provider: S3CredentialProvider | None = None,
        **kwargs: Unpack[S3Config],  # type: ignore # noqa: PGH003 (bucket key overlaps with positional arg)
    ) -> None:
        """Create a new S3Store.

        Args:
            bucket: The AWS bucket to use.

        Keyword Args:
            prefix: A prefix within the bucket to use for all operations.
            config: AWS configuration. Values in this config will override values inferred from the
            environment. Defaults to None.
            client_options: HTTP Client options. Defaults to None.
            retry_config: Retry configuration. Defaults to None.
            credential_provider: A callback to provide custom S3 credentials.
            kwargs: AWS configuration values. Supports the same values as `config`, but as named keyword
            args.

        Returns:
            S3Store

        """
    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        config: S3Config | None = None,
        client_options: ClientConfig | None = None,
        retry_config: RetryConfig | None = None,
        credential_provider: S3CredentialProvider | None = None,
        **kwargs: Unpack[S3Config],
    ) -> Self:
        """Parse available connection info from a well-known storage URL.

        Any path on the URL will be assigned as the `prefix` for the store. So if you
        pass `s3://bucket/path/to/directory`, the store will be created with a prefix of
        `path/to/directory`, and all further operations will use paths relative to that
        prefix.

        The supported url schemes are:

        - `s3://<bucket>/<path>`
        - `s3a://<bucket>/<path>`
        - `https://s3.<region>.amazonaws.com/<bucket>`
        - `https://<bucket>.s3.<region>.amazonaws.com`
        - `https://ACCOUNT_ID.r2.cloudflarestorage.com/bucket`

        Args:
            url: well-known storage URL.

        Keyword Args:
            config: AWS Configuration. Values in this config will override values inferred from the url.
            Defaults to None.
            client_options: HTTP Client options. Defaults to None.
            retry_config: Retry configuration. Defaults to None.
            credential_provider: A callback to provide custom S3 credentials.
            kwargs: AWS configuration values. Supports the same values as `config`, but as named keyword
            args.


        Returns:
            S3Store

        """

    def __eq__(self, value: object) -> bool: ...
    def __getnewargs_ex__(self): ...
    @property
    def prefix(self) -> str | None:
        """Get the prefix applied to all operations in this store, if any."""
    @property
    def config(self) -> S3Config:
        """Get the underlying S3 config parameters."""
    @property
    def client_options(self) -> ClientConfig | None:
        """Get the store's client configuration."""
    @property
    def credential_provider(self) -> S3CredentialProvider | None:
        """Get the store's credential provider."""
    @property
    def retry_config(self) -> RetryConfig | None:
        """Get the store's retry configuration."""
