WITH UserReputation AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        u.Reputation,
        COUNT(DISTINCT b.Id) AS BadgeCount
    FROM Users u
    LEFT JOIN Badges b ON u.Id = b.UserId
    WHERE u.Reputation > 1000
    GROUP BY u.Id, u.DisplayName, u.Reputation
),

PostStatistics AS (
    SELECT 
        p.OwnerUserId,
        COUNT(p.Id) AS TotalPosts,
        SUM(CASE WHEN p.PostTypeId = 1 THEN 1 ELSE 0 END) AS TotalQuestions,
        SUM(CASE WHEN p.PostTypeId = 2 THEN 1 ELSE 0 END) AS TotalAnswers,
        SUM(CASE WHEN p.PostTypeId IN (1, 2, 6, 7) THEN p.Score ELSE 0 END) AS TotalScore,
        AVG(p.ViewCount) AS AvgViews
    FROM Posts p
    GROUP BY p.OwnerUserId
),

CombinedStatistics AS (
    SELECT 
        ur.UserId,
        ur.DisplayName,
        ur.Reputation,
        ur.BadgeCount,
        ps.TotalPosts,
        ps.TotalQuestions,
        ps.TotalAnswers,
        ps.TotalScore,
        ps.AvgViews
    FROM UserReputation ur
    JOIN PostStatistics ps ON ur.UserId = ps.OwnerUserId
)

SELECT 
    UserId,
    DisplayName,
    Reputation,
    BadgeCount,
    TotalPosts,
    TotalQuestions,
    TotalAnswers,
    TotalScore,
    AvgViews,
    CASE 
        WHEN TotalPosts > 100 THEN 'Veteran'
        WHEN TotalPosts > 50 THEN 'Active'
        ELSE 'Newcomer' 
    END AS UserRank
FROM CombinedStatistics
ORDER BY Reputation DESC, TotalScore DESC;
