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

class HfStore:
    """A Hugging Face Hub object store, rooted at one repository and revision.

    Reading an ``hf://`` URL needs no store; this covers a token held in a variable
    rather than the environment, and reads that must stay anonymous regardless of it.

    Construct it and pass it to ``vortex.io.read_url(path, store=hf_store)``, where
    ``path`` is a path within the repository.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        repo_type: str = "dataset",
        revision: str | None = None,
        token: bool | str | None = None,
        endpoint: str | None = None,
    ) -> None: ...
