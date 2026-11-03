"""Optional: upload reports/ to S3 so eval history can be queried with Athena.

    pip install -r requirements-aws.txt
    REPORTS_BUCKET=my-bucket python -m aws.upload_reports [--prefix fancare]

Uploads every file under reports/ to s3://$REPORTS_BUCKET/<prefix>/reports/<run date>/...
and the eval history CSV to s3://$REPORTS_BUCKET/<prefix>/eval_history/eval_history.csv
(the location the Athena table in aws/athena/create_table.sql points at).
Does nothing (exit 0) when AWS credentials or REPORTS_BUCKET are missing.
"""
import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def aws_configured() -> str | None:
    if not os.getenv("REPORTS_BUCKET"):
        return "REPORTS_BUCKET is not set"
    try:
        import boto3
    except ImportError:
        return "boto3 is not installed (pip install -r requirements-aws.txt)"
    if boto3.session.Session().get_credentials() is None:
        return "no AWS credentials found"
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="fancare")
    args = p.parse_args(argv)
    problem = aws_configured()
    if problem:
        print(f"Skipping S3 upload: {problem}")
        return 0

    import boto3

    s3 = boto3.client("s3")
    bucket = os.environ["REPORTS_BUCKET"]
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    count = 0
    for path in REPORTS.rglob("*"):
        if path.is_file():
            key = f"{args.prefix}/reports/{stamp}/{path.relative_to(REPORTS).as_posix()}"
            s3.upload_file(str(path), bucket, key)
            count += 1
    history = REPORTS / "eval_history.csv"
    if history.exists():
        s3.upload_file(str(history), bucket, f"{args.prefix}/eval_history/eval_history.csv")
    print(f"Uploaded {count} files to s3://{bucket}/{args.prefix}/reports/{stamp}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
