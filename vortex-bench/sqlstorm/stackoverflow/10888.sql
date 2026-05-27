WITH PostStats AS (
    SELECT 
        pt.Name AS PostType,
        COUNT(p.Id) AS TotalPosts,
        COUNT(DISTINCT p.OwnerUserId) AS UniqueUsers,
        AVG(u.Reputation) AS AverageUserReputation,
        SUM(p.ViewCount) AS TotalViews,
        SUM(p.Score) AS TotalScore
    FROM 
        Posts p
    JOIN 
        PostTypes pt ON p.PostTypeId = pt.Id
    LEFT JOIN 
        Users u ON p.OwnerUserId = u.Id
    GROUP BY 
        pt.Name
)

SELECT 
    PostType,
    TotalPosts,
    UniqueUsers,
    AverageUserReputation,
    TotalViews,
    TotalScore
FROM 
    PostStats
ORDER BY 
    TotalPosts DESC;