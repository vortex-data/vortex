# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import sys


def main() -> None:
    from vortex._lib.cli import launch

    launch(sys.argv)
