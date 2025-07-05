#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow.dataset

def dataset_from_url(url: str) -> pyarrow.dataset.Dataset: ...
