WITH UserStats AS (
    SELECT 
        U.Id AS UserId,
        U.DisplayName,
        U.Reputation,
        COUNT(DISTINCT P.Id) AS PostCount,
        SUM(CASE WHEN P.ViewCount > 1000 THEN 1 ELSE 0 END) AS PopularPosts,
        SUM(CASE WHEN P.Score > 50 THEN 1 ELSE 0 END) AS HighScorePosts
    FROM Users U
    LEFT JOIN Posts P ON U.Id = P.OwnerUserId
    WHERE U.Reputation > 100
    GROUP BY U.Id, U.DisplayName, U.Reputation
), 
BadgeCounts AS (
    SELECT 
        B.UserId,
        COUNT(*) AS BadgeCount
    FROM Badges B
    GROUP BY B.UserId
), 
Report AS (
    SELECT 
        US.UserId,
        US.DisplayName,
        US.Reputation,
        US.PostCount,
        US.PopularPosts,
        US.HighScorePosts,
        COALESCE(BC.BadgeCount, 0) AS BadgeCount
    FROM UserStats US
    LEFT JOIN BadgeCounts BC ON US.UserId = BC.UserId
)
SELECT 
    R.DisplayName,
    R.Reputation,
    R.PostCount,
    R.PopularPosts,
    R.HighScorePosts,
    R.BadgeCount,
    CASE 
        WHEN R.Reputation > 1000 THEN 'Elite'
        WHEN R.Reputation > 500 THEN 'Pro'
        ELSE 'Novice'
    END AS UserTier
FROM Report R
WHERE R.PostCount > 10
ORDER BY R.Reputation DESC, R.PostCount DESC
LIMIT 20;
