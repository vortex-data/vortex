# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Temporary counterfactual build; leaves registry and production dependency untouched."""
import os
from pathlib import Path
import shutil

cargo_home = Path(os.environ.get('CARGO_HOME', Path.home() / '.cargo'))
sources = list((cargo_home / 'registry/src').glob('*/arrow-select-59.3.2'))
assert len(sources) == 1, sources
target = Path('target/scan-coalescer-arrow-select')
shutil.copytree(sources[0], target, dirs_exist_ok=True)
path = target / 'src/coalesce.rs'
source = path.read_text()
needle = 'if self.buffered_rows > limit {'
assert source.count(needle) == 1
source = source.replace(needle, 'if self.buffered_rows > 0 {')
path.write_text(source)
print('Prepared temporary coalescer experiment:', target)
