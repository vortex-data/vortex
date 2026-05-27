WITH UserPostCounts AS (
    SELECT 
        U.Id AS UserId,
        COUNT(P.Id) AS PostCount,
        SUM(COALESCE(P.Score, 0)) AS TotalScore,
        SUM(COALESCE(P.ViewCount, 0)) AS TotalViews
    FROM Users U
    LEFT JOIN Posts P ON U.Id = P.OwnerUserId
    GROUP BY U.Id
),
PostTags AS (
    SELECT 
        P.Id AS PostId,
        STRING_AGG(T.TagName, ', ') AS Tags
    FROM Posts P
    JOIN Tags T ON P.Tags LIKE '%' || T.TagName || '%' 
    GROUP BY P.Id
),
UserBadges AS (
    SELECT 
        B.UserId,
        COUNT(B.Id) AS BadgeCount
    FROM Badges B
    GROUP BY B.UserId
)
SELECT 
    U.Id,
    U.DisplayName,
    U.Reputation,
    UPC.PostCount,
    UPC.TotalScore,
    UPC.TotalViews,
    T.Tags,
    COALESCE(UB.BadgeCount, 0) AS BadgeCount
FROM Users U
LEFT JOIN UserPostCounts UPC ON U.Id = UPC.UserId
LEFT JOIN PostTags T ON U.Id = T.PostId
LEFT JOIN UserBadges UB ON U.Id = UB.UserId
ORDER BY UPC.PostCount DESC, UPC.TotalScore DESC
LIMIT 100;