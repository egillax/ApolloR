{DEFAULT @partitions = 200}

DROP TABLE IF EXISTS #partition_table;

SELECT row_id AS observation_period_id,
  observation_period.person_id,
  CASE 
    WHEN CAST(DATEADD(DAY, @window_start, cohort_start_date) AS DATE) < observation_period_start_date THEN observation_period_start_date
    ELSE CAST(DATEADD(DAY, @window_start, cohort_start_date) AS DATE)
  END AS observation_period_start_date,
  CAST(DATEADD(DAY, @window_end, cohort_start_date) AS DATE) AS observation_period_end_date,
  rn % @partitions + 1 AS partition_id
INTO #partition_table
FROM (
  SELECT @row_id_field AS row_id, 
    subject_id,
    ROW_NUMBER() OVER (ORDER BY NEWID()) AS rn,
    cohort_start_date
  FROM @cohort_table
  {@cohort_id != -1} ? {WHERE cohort_definition_id IN (@cohort_id)}
) cohort
INNER JOIN @cdm_database_schema.observation_period
  ON subject_id = person_id
    AND cohort_start_date >= observation_period_start_date
    AND cohort_start_date <= observation_period_end_date;
