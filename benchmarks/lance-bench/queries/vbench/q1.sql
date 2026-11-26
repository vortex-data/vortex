-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

SELECT id, cosine_distance(emb, $1) AS distance
FROM recipes
ORDER BY distance
LIMIT 10
