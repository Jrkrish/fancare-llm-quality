-- External table over the eval history uploaded by aws/upload_reports.py.
-- Replace <bucket> and <prefix>. Columns match eval/report.py HISTORY_COLUMNS.
CREATE EXTERNAL TABLE IF NOT EXISTS fancare_eval_history (
  `timestamp`            string,
  run_id                 string,
  profile                string,
  prompt_version         string,
  provider               string,
  model                  string,
  retriever_mode         string,
  rerank                 string,
  judge                  string,
  runs_per_case          int,
  n_cases                double,
  n_runs                 double,
  faithfulness           double,
  answer_relevancy       double,
  contextual_precision   double,
  contextual_recall      double,
  citation_correctness   double,
  refusal_correctness    double,
  retrieval_recall       double,
  hallucination_rate     double,
  case_pass_rate         double,
  error_rate             double,
  unstable_cases         double,
  latency_p50_ms         double,
  latency_p95_ms         double,
  cost_usd_total         double,
  gates_passed           string
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '"')
LOCATION 's3://<bucket>/<prefix>/eval_history/'
TBLPROPERTIES ('skip.header.line.count' = '1');
