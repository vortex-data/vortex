SELECT
    p.Id AS PostId,
    p.Title,
    p.CreationDate,
    u.DisplayName AS OwnerDisplayName,
    COUNT(c.Id) AS CommentCount,
    COUNT(v.Id) AS VoteCount
FROM
    Posts p
JOIN
    Users u ON p.OwnerUserId = u.Id
LEFT JOIN
    Comments c ON p.Id = c.PostId
LEFT JOIN
    Votes v ON p.Id = v.PostId
WHERE
    p.PostTypeId = 1 
GROUP BY
    p.Id, p.Title, p.CreationDate, u.DisplayName
ORDER BY
    p.CreationDate DESC
LIMIT 10;