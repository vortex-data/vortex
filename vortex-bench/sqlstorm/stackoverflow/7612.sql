
WITH UserStats AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        u.Reputation,
        COUNT(DISTINCT p.Id) AS PostCount,
        SUM(CASE WHEN p.PostTypeId = 1 THEN 1 ELSE 0 END) AS QuestionCount,
        SUM(CASE WHEN p.PostTypeId = 2 THEN 1 ELSE 0 END) AS AnswerCount,
        SUM(CASE WHEN p.PostTypeId IN (10, 11, 12) THEN 1 ELSE 0 END) AS ClosedPosts,
        SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 ELSE 0 END) AS UpVotes,
        SUM(CASE WHEN v.VoteTypeId = 3 THEN 1 ELSE 0 END) AS DownVotes
    FROM Users u
    LEFT JOIN Posts p ON u.Id = p.OwnerUserId
    LEFT JOIN Votes v ON p.Id = v.PostId
    GROUP BY u.Id, u.DisplayName, u.Reputation
), BadgeStats AS (
    SELECT 
        b.UserId,
        COUNT(b.Id) AS BadgeCount,
        COUNT(CASE WHEN b.Class = 1 THEN 1 END) AS GoldBadges,
        COUNT(CASE WHEN b.Class = 2 THEN 1 END) AS SilverBadges,
        COUNT(CASE WHEN b.Class = 3 THEN 1 END) AS BronzeBadges
    FROM Badges b
    GROUP BY b.UserId
), PostHistoryStats AS (
    SELECT 
        ph.UserId,
        COUNT(ph.Id) AS EditCount,
        SUM(CASE WHEN ph.PostHistoryTypeId IN (4, 5, 6) THEN 1 ELSE 0 END) AS TitleEdits,
        SUM(CASE WHEN ph.PostHistoryTypeId IN (10, 11) THEN 1 ELSE 0 END) AS ClosedPostChanges
    FROM PostHistory ph
    GROUP BY ph.UserId
)
SELECT 
    us.UserId,
    us.DisplayName,
    us.Reputation,
    us.PostCount,
    us.QuestionCount,
    us.AnswerCount,
    us.ClosedPosts,
    us.UpVotes,
    us.DownVotes,
    bs.BadgeCount,
    bs.GoldBadges,
    bs.SilverBadges,
    bs.BronzeBadges,
    phs.EditCount,
    phs.TitleEdits,
    phs.ClosedPostChanges
FROM UserStats us
LEFT JOIN BadgeStats bs ON us.UserId = bs.UserId
LEFT JOIN PostHistoryStats phs ON us.UserId = phs.UserId
WHERE us.Reputation > 1000
ORDER BY us.Reputation DESC, us.PostCount DESC
LIMIT 50;
