# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Keep GitHub OIDC tokens fresh for object_store's renewable AWS provider."""

import json
import os
import threading
import urllib.parse
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def environment(role_arn: str | None) -> Iterator[dict[str, str]]:
    env = os.environ.copy()
    if not role_arn:
        yield env
        return
    url = env["ACTIONS_ID_TOKEN_REQUEST_URL"]
    separator = "&" if "?" in url else "?"
    url += separator + urllib.parse.urlencode({"audience": "sts.amazonaws.com"})
    with TemporaryDirectory(prefix="s3-validation-oidc-") as directory:
        path = Path(directory) / "token"

        def refresh():
            request = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {env['ACTIONS_ID_TOKEN_REQUEST_TOKEN']}"}
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                token = json.load(response)["value"]
            if not isinstance(token, str) or not token:
                raise ValueError("OIDC response has no token")
            temporary = path.with_suffix(".new")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w") as output:
                output.write(token)
            temporary.replace(path)

        refresh()
        # Static credentials take precedence over web identity in object_store.
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"):
            env.pop(name, None)
        env.update(
            AWS_WEB_IDENTITY_TOKEN_FILE=str(path),
            AWS_ROLE_ARN=role_arn,
            AWS_ROLE_SESSION_NAME="GitHubActions",
        )
        stopped = threading.Event()

        def renew():
            while not stopped.wait(120):
                try:
                    refresh()
                except (OSError, ValueError, KeyError) as error:
                    # Avoid logging an HTTP response or URL that might contain credentials.
                    print(f"OIDC refresh failed ({type(error).__name__}); retrying", flush=True)

        thread = threading.Thread(target=renew, name="s3-validation-oidc", daemon=True)
        thread.start()
        try:
            yield env
        finally:
            stopped.set()
            thread.join(timeout=30)
