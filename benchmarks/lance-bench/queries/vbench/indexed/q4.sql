-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

SELECT id, distance
FROM vector_search('$table', 'emb', '$vector', 10)
WHERE price >= 90
ORDER BY distance
