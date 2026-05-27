
SELECT 
    u.DisplayName, 
    COUNT(p.Id) AS NumberOfPosts, 
    SUM(CASE WHEN p.PostTypeId = 1 THEN 1 ELSE 0 END) AS Questions, 
    SUM(CASE WHEN p.PostTypeId = 2 THEN 1 ELSE 0 END) AS Answers 
FROM 
    Users u 
LEFT JOIN 
    Posts p ON u.Id = p.OwnerUserId 
GROUP BY 
    u.DisplayName, u.Id 
ORDER BY 
    NumberOfPosts DESC 
LIMIT 10;
