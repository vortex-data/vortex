
WITH RECURSIVE UserBadgeCounts AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        SUM(CASE WHEN b.Class = 1 THEN 1 ELSE 0 END) AS GoldBadgeCount,
        SUM(CASE WHEN b.Class = 2 THEN 1 ELSE 0 END) AS SilverBadgeCount,
        SUM(CASE WHEN b.Class = 3 THEN 1 ELSE 0 END) AS BronzeBadgeCount
    FROM 
        Users u
    LEFT JOIN 
        Badges b ON u.Id = b.UserId
    GROUP BY 
        u.Id, u.DisplayName
),
PostMetrics AS (
    SELECT
        p.OwnerUserId,
        COUNT(DISTINCT p.Id) AS TotalPosts,
        COUNT(DISTINCT CASE WHEN p.PostTypeId = 1 THEN p.Id END) AS TotalQuestions,
        COUNT(DISTINCT CASE WHEN p.PostTypeId = 2 THEN p.Id END) AS TotalAnswers,
        SUM(p.Score) AS TotalScore,
        AVG(p.ViewCount) AS AvgViewCount,
        MAX(p.CreationDate) AS LastPostDate
    FROM 
        Posts p
    GROUP BY 
        p.OwnerUserId
),
CombinedMetrics AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        COALESCE(ubc.GoldBadgeCount, 0) AS GoldBadgeCount,
        COALESCE(ubc.SilverBadgeCount, 0) AS SilverBadgeCount,
        COALESCE(ubc.BronzeBadgeCount, 0) AS BronzeBadgeCount,
        COALESCE(pm.TotalPosts, 0) AS TotalPosts,
        COALESCE(pm.TotalQuestions, 0) AS TotalQuestions,
        COALESCE(pm.TotalAnswers, 0) AS TotalAnswers,
        COALESCE(pm.TotalScore, 0) AS TotalScore,
        COALESCE(pm.AvgViewCount, 0) AS AvgViewCount,
        pm.LastPostDate
    FROM 
        Users u
    LEFT JOIN 
        UserBadgeCounts ubc ON u.Id = ubc.UserId
    LEFT JOIN 
        PostMetrics pm ON u.Id = pm.OwnerUserId
)
SELECT 
    c.UserId,
    c.DisplayName,
    c.GoldBadgeCount,
    c.SilverBadgeCount,
    c.BronzeBadgeCount,
    c.TotalPosts,
    c.TotalQuestions,
    c.TotalAnswers,
    c.TotalScore,
    c.AvgViewCount,
    CASE 
        WHEN c.LastPostDate IS NOT NULL THEN DATE '2024-10-01' - c.LastPostDate 
        ELSE NULL 
    END AS DaysSinceLastPost
FROM 
    CombinedMetrics c
WHERE 
    (c.TotalQuestions > 0 OR c.TotalAnswers > 0)
    AND (c.GoldBadgeCount > 0 OR c.SilverBadgeCount > 0 OR c.BronzeBadgeCount > 0)
ORDER BY 
    c.TotalScore DESC,
    c.TotalPosts DESC
LIMIT 100;
