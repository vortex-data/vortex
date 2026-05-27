
SELECT 
    p.Id as PostId,
    p.Title,
    p.CreationDate,
    u.DisplayName as OwnerDisplayName,
    p.Score,
    p.ViewCount,
    COUNT(c.Id) as CommentCount
FROM 
    Posts p
JOIN 
    Users u ON p.OwnerUserId = u.Id
LEFT JOIN 
    Comments c ON p.Id = c.PostId
WHERE 
    p.PostTypeId = 1 
GROUP BY 
    p.Id, p.Title, p.CreationDate, u.DisplayName, p.Score, p.ViewCount
ORDER BY 
    p.CreationDate DESC
LIMIT 10;
