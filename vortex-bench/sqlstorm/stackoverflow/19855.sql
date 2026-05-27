SELECT P.Id, P.Title, P.CreationDate, U.DisplayName, P.Score 
FROM Posts P
JOIN Users U ON P.OwnerUserId = U.Id
WHERE P.PostTypeId = 1 
ORDER BY P.CreationDate DESC
LIMIT 10;