"""Optional AWS checks: skipped unless AWS credentials and REPORTS_BUCKET are configured."""
import os

import pytest

pytestmark = pytest.mark.aws
boto3 = pytest.importorskip("boto3", reason="install requirements-aws.txt to run AWS tests")

if not os.getenv("REPORTS_BUCKET") or boto3.session.Session().get_credentials() is None:
    pytest.skip("AWS credentials or REPORTS_BUCKET not set", allow_module_level=True)


def test_upload_reports_to_s3(tmp_path):
    from aws.upload_reports import main

    assert main(["--prefix", "fancare-ci-test"]) == 0
    s3 = boto3.client("s3")
    listed = s3.list_objects_v2(Bucket=os.environ["REPORTS_BUCKET"], Prefix="fancare-ci-test/")
    assert listed.get("KeyCount", 0) > 0
