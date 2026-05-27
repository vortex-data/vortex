
WITH RecentActivity AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        COUNT(DISTINCT p.Id) AS TotalPosts,
        SUM(CASE WHEN p.PostTypeId = 1 THEN 1 ELSE 0 END) AS QuestionsCount,
        SUM(CASE WHEN p.PostTypeId = 2 THEN 1 ELSE 0 END) AS AnswersCount,
        SUM(CASE WHEN p.PostTypeId IN (3, 4, 5) THEN 1 ELSE 0 END) AS WikiPostsCount,
        SUM(v.BountyAmount) AS TotalBounty,
        RANK() OVER (PARTITION BY u.Id ORDER BY SUM(v.BountyAmount) DESC) AS BountyRank
    FROM Users u
    LEFT JOIN Posts p ON u.Id = p.OwnerUserId 
    LEFT JOIN Votes v ON p.Id = v.PostId AND v.VoteTypeId IN (8, 9)  
    WHERE u.Reputation > 1000 AND u.Location IS NOT NULL 
    GROUP BY u.Id, u.DisplayName
),
TopUsers AS (
    SELECT 
        UserId,
        DisplayName,
        TotalPosts,
        QuestionsCount,
        AnswersCount,
        WikiPostsCount,
        TotalBounty,
        BountyRank
    FROM RecentActivity
    WHERE BountyRank <= 5  
),
UserBadges AS (
    SELECT 
        b.UserId,
        COUNT(CASE WHEN b.Class = 1 THEN 1 END) AS GoldBadges,
        COUNT(CASE WHEN b.Class = 2 THEN 1 END) AS SilverBadges,
        COUNT(CASE WHEN b.Class = 3 THEN 1 END) AS BronzeBadges
    FROM Badges b
    GROUP BY b.UserId
)
SELECT 
    tu.DisplayName,
    tu.TotalPosts,
    tu.QuestionsCount,
    tu.AnswersCount,
    tu.WikiPostsCount,
    COALESCE(ub.GoldBadges, 0) AS GoldBadges,
    COALESCE(ub.SilverBadges, 0) AS SilverBadges,
    COALESCE(ub.BronzeBadges, 0) AS BronzeBadges,
    tu.TotalBounty
FROM TopUsers tu
LEFT JOIN UserBadges ub ON tu.UserId = ub.UserId
ORDER BY tu.TotalBounty DESC, tu.DisplayName ASC;
