-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

SELECT id, distance
FROM recipe_knn('$vector', 10)
ORDER BY distance
