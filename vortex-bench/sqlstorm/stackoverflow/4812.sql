WITH MostActiveUsers AS (
    SELECT u.Id, u.DisplayName, COUNT(p.Id) AS PostCount, SUM(COALESCE(p.ViewCount, 0)) AS TotalViews
    FROM Users u
    JOIN Posts p ON u.Id = p.OwnerUserId
    WHERE u.Reputation > 1000
    GROUP BY u.Id, u.DisplayName
), UserBadges AS (
    SELECT b.UserId, COUNT(b.Id) AS BadgeCount, MAX(b.Class) AS HighestBadgeClass
    FROM Badges b
    GROUP BY b.UserId
), UserPostStats AS (
    SELECT ua.Id, ua.DisplayName, ua.PostCount, ua.TotalViews,
           COALESCE(ub.BadgeCount, 0) AS BadgeCount,
           COALESCE(ub.HighestBadgeClass, 0) AS HighestBadgeClass
    FROM MostActiveUsers ua
    LEFT JOIN UserBadges ub ON ua.Id = ub.UserId
), RecentPosts AS (
    SELECT p.Id, p.Title, p.CreationDate, p.OwnerUserId,
           ROW_NUMBER() OVER (PARTITION BY p.OwnerUserId ORDER BY p.CreationDate DESC) AS rn
    FROM Posts p
    WHERE p.CreationDate > cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '30 days'
)
SELECT ups.DisplayName, ups.PostCount, ups.TotalViews, ups.BadgeCount,
       ups.HighestBadgeClass, rp.Title AS LatestPostTitle, rp.CreationDate AS LatestPostDate
FROM UserPostStats ups
LEFT JOIN RecentPosts rp ON ups.Id = rp.OwnerUserId AND rp.rn = 1
WHERE ups.TotalViews > 100
ORDER BY ups.PostCount DESC, ups.TotalViews DESC, ups.BadgeCount DESC;