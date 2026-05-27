SELECT 
    PH.PostId,
    COUNT(PH.Id) AS RevisionCount,
    MIN(PH.CreationDate) AS FirstRevisionDate,
    MAX(PH.CreationDate) AS LastRevisionDate,
    U.DisplayName AS LastEditedBy,
    P.Title,
    P.Score,
    P.ViewCount,
    P.AnswerCount,
    P.CommentCount
FROM 
    PostHistory PH
JOIN 
    Posts P ON PH.PostId = P.Id
LEFT JOIN 
    Users U ON PH.UserId = U.Id
GROUP BY 
    PH.PostId, U.DisplayName, P.Title, P.Score, P.ViewCount, P.AnswerCount, P.CommentCount
ORDER BY 
    RevisionCount DESC
LIMIT 10;
