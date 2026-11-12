# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Create the reproducible 16 MiB probe shared by S3 measurement and calibration runs."""

import random
from pathlib import Path


def main():
    Path("s3-probe.bin").write_bytes(random.Random(20261028).randbytes(16 * 1024 * 1024))


if __name__ == "__main__":
    main()
