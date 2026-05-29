WITH UserPostStats AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        COUNT(p.Id) AS PostCount,
        SUM(CASE WHEN p.PostTypeId = 1 THEN 1 ELSE 0 END) AS QuestionCount,
        SUM(CASE WHEN p.PostTypeId = 2 THEN 1 ELSE 0 END) AS AnswerCount,
        SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 ELSE 0 END) AS UpVoteCount,
        SUM(CASE WHEN v.VoteTypeId = 3 THEN 1 ELSE 0 END) AS DownVoteCount
    FROM 
        Users u
    LEFT JOIN 
        Posts p ON u.Id = p.OwnerUserId
    LEFT JOIN 
        Votes v ON p.Id = v.PostId
    GROUP BY 
        u.Id, u.DisplayName
),
UserBadgeStats AS (
    SELECT 
        UserId,
        COUNT(Id) AS BadgeCount,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS GoldBadgeCount,
        SUM(CASE WHEN Class = 2 THEN 1 ELSE 0 END) AS SilverBadgeCount,
        SUM(CASE WHEN Class = 3 THEN 1 ELSE 0 END) AS BronzeBadgeCount
    FROM 
        Badges
    GROUP BY 
        UserId
)
SELECT 
    u.Id AS UserId,
    u.DisplayName,
    COALESCE(ups.PostCount, 0) AS TotalPosts,
    COALESCE(ups.QuestionCount, 0) AS TotalQuestions,
    COALESCE(ups.AnswerCount, 0) AS TotalAnswers,
    COALESCE(ups.UpVoteCount, 0) AS TotalUpVotes,
    COALESCE(ups.DownVoteCount, 0) AS TotalDownVotes,
    COALESCE(ubs.BadgeCount, 0) AS TotalBadges,
    COALESCE(ubs.GoldBadgeCount, 0) AS TotalGoldBadges,
    COALESCE(ubs.SilverBadgeCount, 0) AS TotalSilverBadges,
    COALESCE(ubs.BronzeBadgeCount, 0) AS TotalBronzeBadges
FROM 
    Users u
LEFT JOIN 
    UserPostStats ups ON u.Id = ups.UserId
LEFT JOIN 
    UserBadgeStats ubs ON u.Id = ubs.UserId
ORDER BY 
    TotalPosts DESC
LIMIT 100;