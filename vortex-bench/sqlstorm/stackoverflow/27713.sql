
WITH UserActivity AS (
    SELECT 
        U.Id AS UserId,
        U.DisplayName, 
        COUNT(P.Id) AS TotalPosts,
        SUM(CASE WHEN P.PostTypeId = 1 THEN 1 ELSE 0 END) AS QuestionsPosted,
        SUM(CASE WHEN P.PostTypeId = 2 THEN 1 ELSE 0 END) AS AnswersPosted,
        SUM(CASE WHEN P.ViewCount > 100 THEN 1 ELSE 0 END) AS PopularPostsCount,
        COALESCE(SUM(CASE WHEN B.Class = 1 THEN 1 ELSE 0 END), 0) AS GoldBadges,
        COALESCE(SUM(CASE WHEN B.Class = 2 THEN 1 ELSE 0 END), 0) AS SilverBadges,
        COALESCE(SUM(CASE WHEN B.Class = 3 THEN 1 ELSE 0 END), 0) AS BronzeBadges
    FROM 
        Users U
    LEFT JOIN 
        Posts P ON U.Id = P.OwnerUserId
    LEFT JOIN 
        Badges B ON U.Id = B.UserId
    GROUP BY 
        U.Id, U.DisplayName
),
UserPostHistory AS (
    SELECT 
        UP.Id AS UserId,
        COUNT(PH.Id) AS TotalPostEdits,
        SUM(CASE WHEN PH.PostHistoryTypeId IN (4, 5, 6) THEN 1 ELSE 0 END) AS TitleBodyTagEdits,
        SUM(CASE WHEN PH.PostHistoryTypeId = 10 THEN 1 ELSE 0 END) AS CloseVotes,
        SUM(CASE WHEN PH.PostHistoryTypeId = 11 THEN 1 ELSE 0 END) AS ReopenVotes
    FROM 
        Users UP
    LEFT JOIN 
        PostHistory PH ON UP.Id = PH.UserId
    GROUP BY 
        UP.Id
),
UserStatistics AS (
    SELECT 
        UA.UserId,
        UA.DisplayName,
        UA.TotalPosts,
        UA.QuestionsPosted,
        UA.AnswersPosted,
        UA.PopularPostsCount,
        UA.GoldBadges,
        UA.SilverBadges,
        UA.BronzeBadges,
        UPH.TotalPostEdits,
        UPH.TitleBodyTagEdits,
        UPH.CloseVotes,
        UPH.ReopenVotes
    FROM 
        UserActivity UA
    LEFT JOIN 
        UserPostHistory UPH ON UA.UserId = UPH.UserId
)
SELECT 
    UserId,
    DisplayName,
    TotalPosts,
    QuestionsPosted,
    AnswersPosted,
    PopularPostsCount,
    GoldBadges,
    SilverBadges,
    BronzeBadges,
    TotalPostEdits,
    TitleBodyTagEdits,
    CloseVotes,
    ReopenVotes,
    ROUND(COALESCE(TotalPosts::DECIMAL / NULLIF(QuestionsPosted, 0), 0), 2) AS PostToQuestionRatio,
    ROUND(COALESCE(AnswersPosted::DECIMAL / NULLIF(QuestionsPosted, 0), 0), 2) AS AnswerToQuestionRatio
FROM 
    UserStatistics
ORDER BY 
    TotalPosts DESC;
