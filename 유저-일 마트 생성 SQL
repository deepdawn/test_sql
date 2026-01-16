-- 1. 유저 활동 집계 (Activity)
WITH user_activity AS (
    SELECT 
        user_id,
        DATE(event_time) AS date,
        -- 추후 조인을 위한 키
        MAX(country) as country, 
        MAX(platform) as platform,
        COUNT(*) AS total_events,
        SUM(CASE WHEN event_name = 'ad_watch' THEN 1 ELSE 0 END) AS ad_watch_count, -- 광고 시청 이벤트가 있다고 가정
        -- 걸음 수 추출 (JSON 파싱 가정: event_properties 내 step_count)
        SUM(COALESCE(CAST(get_json_object(event_properties, '$.step_count') AS INT), 0)) AS total_steps
    FROM fact_app_events
    WHERE date(event_time) = '${TARGET_DATE}'
    GROUP BY user_id, DATE(event_time)
),

-- 2. 포인트 적립 집계 (Points)
user_points AS (
    SELECT 
        user_id,
        DATE(created_at) AS date,
        SUM(CASE WHEN point_delta > 0 THEN point_delta ELSE 0 END) AS points_earned
    FROM fact_points_transaction
    WHERE date(created_at) = '${TARGET_DATE}'
    GROUP BY user_id, DATE(created_at)
),

-- 3. 세그먼트별 광고 매출 집계 (Revenue Segment)
daily_segment_revenue AS (
    SELECT 
        date,
        country,
        platform,
        SUM(revenue) AS total_segment_revenue
    FROM fact_daily_ad_revenue
    WHERE date = '${TARGET_DATE}'
    GROUP BY date, country, platform
),

-- 4. 세그먼트별 DAU 집계 (Revenue Allocation을 위함)
daily_segment_dau AS (
    SELECT
        DATE(event_time) as date,
        country,
        platform,
        COUNT(DISTINCT user_id) as segment_dau
    FROM fact_app_events
    WHERE date(event_time) = '${TARGET_DATE}'
    GROUP BY 1, 2, 3
)

-- 5. 최종 결합 (Final Mart)
INSERT OVERWRITE TABLE mart_user_daily_activity PARTITION (date='${TARGET_DATE}')
SELECT 
    u.user_id,
    u.total_events,
    u.ad_watch_count,
    u.total_steps,
    COALESCE(p.points_earned, 0) AS points_earned,
    -- 광고 매출 배분 로직: (해당 세그먼트 총 매출) / (해당 세그먼트 총 DAU)
    COALESCE(r.total_segment_revenue / NULLIF(d.segment_dau, 0), 0) AS estimated_ad_revenue
FROM 
    user_activity u
LEFT JOIN 
    user_points p ON u.user_id = p.user_id AND u.date = p.date
LEFT JOIN 
    daily_segment_revenue r ON u.date = r.date AND u.country = r.country AND u.platform = r.platform
LEFT JOIN 
    daily_segment_dau d ON u.date = d.date AND u.country = d.country AND u.platform = d.platform;
