-- Sample Athena query: weekly quality and latency trend per prompt version and model,
-- plus how often the gates failed. (OpenCSVSerde reads every column as string, hence the casts.)
SELECT
  date_trunc('week', from_iso8601_timestamp("timestamp"))      AS week,
  prompt_version,
  model,
  count(*)                                                       AS runs,
  round(avg(CAST(faithfulness AS double)), 3)                    AS avg_faithfulness,
  round(avg(CAST(refusal_correctness AS double)), 3)             AS avg_refusal_correctness,
  round(avg(CAST(hallucination_rate AS double)), 3)              AS avg_hallucination_rate,
  round(approx_percentile(CAST(latency_p95_ms AS double), 0.5))  AS median_p95_latency_ms,
  sum(CASE WHEN gates_passed = 'False' THEN 1 ELSE 0 END)        AS failed_gate_runs
FROM fancare_eval_history
WHERE profile = 'full'
GROUP BY 1, 2, 3
ORDER BY week DESC, prompt_version, model;
