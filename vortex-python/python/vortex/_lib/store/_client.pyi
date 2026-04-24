# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2024 Development Seed

from datetime import timedelta
from typing import TypedDict

class ClientConfig(TypedDict, total=False):
    """HTTP client configuration.

    For timeout values (`connect_timeout`, `http2_keep_alive_timeout`,
    `pool_idle_timeout`, and `timeout`), values can either be Python `timedelta`
    objects, or they can be "human-readable duration strings".

    The human-readable duration string is a concatenation of time spans. Where each time
    span is an integer number and a suffix. Supported suffixes:

    - `nsec`, `ns` -- nanoseconds
    - `usec`, `us` -- microseconds
    - `msec`, `ms` -- milliseconds
    - `seconds`, `second`, `sec`, `s`
    - `minutes`, `minute`, `min`, `m`
    - `hours`, `hour`, `hr`, `h`
    - `days`, `day`, `d`
    - `weeks`, `week`, `w`
    - `months`, `month`, `M` -- defined as 30.44 days
    - `years`, `year`, `y` -- defined as 365.25 days

    For example:

    - `"2h 37min"`
    - `"32ms"`

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING` block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from vortex.store import ClientConfig
        ```
    """

    allow_http: bool
    """Allow non-TLS, i.e. non-HTTPS connections."""
    allow_invalid_certificates: bool
    """Skip certificate validation on https connections.

    !!! warning

        You should think very carefully before using this method. If
        invalid certificates are trusted, *any* certificate for *any* site
        will be trusted for use. This includes expired certificates. This
        introduces significant vulnerabilities, and should only be used
        as a last resort or for testing
    """
    connect_timeout: str | timedelta
    """Timeout for only the connect phase of a Client"""
    default_content_type: str
    """Default `CONTENT_TYPE` for uploads"""
    default_headers: dict[str, str] | dict[str, bytes]
    """Default headers to be sent with each request"""
    http1_only: bool
    """Only use http1 connections."""
    http2_keep_alive_interval: str
    """Interval for HTTP2 Ping frames should be sent to keep a connection alive."""
    http2_keep_alive_timeout: str | timedelta
    """Timeout for receiving an acknowledgement of the keep-alive ping."""
    http2_keep_alive_while_idle: str
    """Enable HTTP2 keep alive pings for idle connections"""
    http2_only: bool
    """Only use http2 connections"""
    pool_idle_timeout: str | timedelta
    """The pool max idle timeout.

    This is the length of time an idle connection will be kept alive.
    """
    pool_max_idle_per_host: str
    """Maximum number of idle connections per host."""
    proxy_url: str
    """HTTP proxy to use for requests."""
    timeout: str | timedelta
    """Request timeout.

    The timeout is applied from when the request starts connecting until the
    response body has finished.
    """
    user_agent: str
    """User-Agent header to be used by this client."""
