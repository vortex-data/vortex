#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

class CosStore:
    """A Tencent Cloud COS object store, backed by OpenDAL.

    Construct it with explicit configuration and pass it to
    ``vortex.io.read_url(url, store=cos_store)`` /
    ``vortex.io.write(arrays, path, store=cos_store)``.

    This class is only available when Vortex is built with the ``opendal`` feature.
    """

    def __init__(
        self,
        bucket: str,
        endpoint: str,
        *,
        secret_id: str | None = None,
        secret_key: str | None = None,
        root: str | None = None,
        disable_config_load: bool = False,
    ) -> None: ...


class GoosefsStore:
    """A Tencent Cloud GooseFS object store, backed by OpenDAL.

    Construct it with explicit configuration and pass it to
    ``vortex.io.read_url(url, store=goosefs_store)`` /
    ``vortex.io.write(arrays, path, store=goosefs_store)``.

    This class is only available when Vortex is built with the ``opendal`` feature.
    """

    def __init__(
        self,
        master_addr: str,
        *,
        root: str | None = None,
        block_size: int | None = None,
        chunk_size: int | None = None,
        write_type: str | None = None,
        auth_type: str | None = None,
        auth_username: str | None = None,
    ) -> None: ...
