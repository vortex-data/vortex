from datetime import timedelta
from typing import TypedDict

class BackoffConfig(TypedDict, total=False):
    """Exponential backoff with jitter.

    See <https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/>

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from obstore.store import BackoffConfig
        ```
    """

    init_backoff: timedelta
    """The initial backoff duration.

    Defaults to 100 milliseconds.
    """

    max_backoff: timedelta
    """The maximum backoff duration.

    Defaults to 15 seconds.
    """

    base: int | float
    """The base of the exponential to use.

    Defaults to `2`.
    """

class RetryConfig(TypedDict, total=False):
    """The configuration for how to respond to request errors.

    The following categories of error will be retried:

    * 5xx server errors
    * Connection errors
    * Dropped connections
    * Timeouts for [safe] / read-only requests

    Requests will be retried up to some limit, using exponential
    backoff with jitter. See [`BackoffConfig`][obstore.store.BackoffConfig] for
    more information

    [safe]: https://datatracker.ietf.org/doc/html/rfc7231#section-4.2.1

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from obstore.store import RetryConfig
        ```
    """

    backoff: BackoffConfig
    """The backoff configuration.

    Defaults to the values listed above if not provided.
    """

    max_retries: int
    """
    The maximum number of times to retry a request

    Set to 0 to disable retries.

    Defaults to 10.
    """

    retry_timeout: timedelta
    """
    The maximum length of time from the initial request
    after which no further retries will be attempted

    This not only bounds the length of time before a server
    error will be surfaced to the application, but also bounds
    the length of time a request's credentials must remain valid.

    As requests are retried without renewing credentials or
    regenerating request payloads, this number should be kept
    below 5 minutes to avoid errors due to expired credentials
    and/or request payloads.

    Defaults to 3 minutes.
    """
