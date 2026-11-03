"""Optional: run an Athena query file and print the result rows.

    ATHENA_DATABASE=default ATHENA_OUTPUT=s3://my-bucket/athena-results/ \
        python -m aws.run_athena_query aws/athena/eval_trends.sql
"""
import os
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    missing = [v for v in ("ATHENA_DATABASE", "ATHENA_OUTPUT") if not os.getenv(v)]
    try:
        import boto3
    except ImportError:
        missing.append("boto3")
    if missing:
        print(f"Skipping Athena query: missing {missing}")
        return 0

    athena = boto3.client("athena")
    qid = athena.start_query_execution(
        QueryString=Path(argv[0]).read_text(encoding="utf-8"),
        QueryExecutionContext={"Database": os.environ["ATHENA_DATABASE"]},
        ResultConfiguration={"OutputLocation": os.environ["ATHENA_OUTPUT"]},
    )["QueryExecutionId"]
    while True:
        state = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)
    if state != "SUCCEEDED":
        print(f"Query {state}")
        return 1
    for row in athena.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]:
        print(" | ".join(c.get("VarCharValue", "") for c in row["Data"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
