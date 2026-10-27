# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import io
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import aws_credentials


class CredentialTests(unittest.TestCase):
    def test_renewable_provider_replaces_static_credentials_only_for_child(self):
        env = {
            "ACTIONS_ID_TOKEN_REQUEST_URL": "https://example.invalid/token?request=1",
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "test-bearer",
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_SESSION_TOKEN": "test-session",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b'{"value":"test-token"}')) as request:
                with aws_credentials.environment("test-role") as child:
                    self.assertNotIn("AWS_ACCESS_KEY_ID", child)
                    self.assertNotIn("AWS_SECRET_ACCESS_KEY", child)
                    self.assertNotIn("AWS_SESSION_TOKEN", child)
                    self.assertEqual(child["AWS_ROLE_ARN"], "test-role")
                    path = Path(child["AWS_WEB_IDENTITY_TOKEN_FILE"])
                    self.assertEqual(path.read_text(), "test-token")
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                    self.assertIn("audience=sts.amazonaws.com", request.call_args.args[0].full_url)
                self.assertFalse(path.exists())
            self.assertEqual(os.environ["AWS_ACCESS_KEY_ID"], "test-key")


if __name__ == "__main__":
    unittest.main()
