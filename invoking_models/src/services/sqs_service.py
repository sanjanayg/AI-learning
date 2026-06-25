import json
import logging
import boto3
from config import settings

logger = logging.getLogger(__name__)


def _get_sqs_client():
    return boto3.client(
        "sqs",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


def publish_upload_job(
    job_id: str,
    chat_id: str,
    file_id: str,
    file_name: str,
    file_type: str,
    storage_path: str,
) -> None:
    """
    Publishes a job message to SQS.
    Only metadata + storage_path is sent — never raw file bytes.
    Message size stays well under the 256KB SQS limit.
    """
    message = {
        "job_id": job_id,
        "chat_id": chat_id,
        "file_id": file_id,
        "file_name": file_name,
        "file_type": file_type,       # MIME type — worker uses this for routing
        "storage_path": storage_path,
    }
    client = _get_sqs_client()
    client.send_message(
        QueueUrl=settings.SQS_QUEUE_URL,
        MessageBody=json.dumps(message),
    )
    logger.info("Published SQS job: job_id=%s file=%s", job_id, file_name)


def receive_messages(max_messages: int = 1, wait_seconds: int = 20) -> list[dict]:
    """
    Long-polls SQS for up to `wait_seconds` (max 20 per AWS limit).
    Returns list of raw SQS message dicts (Body + ReceiptHandle).
    """
    client = _get_sqs_client()
    response = client.receive_message(
        QueueUrl=settings.SQS_QUEUE_URL,
        MaxNumberOfMessages=max_messages,
        WaitTimeSeconds=wait_seconds,
        AttributeNames=["All"],
    )
    return response.get("Messages", [])


def delete_message(receipt_handle: str) -> None:
    """
    Deletes a message from the queue after successful processing.
    Only called on success — failures leave the message in-flight to be retried.
    """
    client = _get_sqs_client()
    client.delete_message(
        QueueUrl=settings.SQS_QUEUE_URL,
        ReceiptHandle=receipt_handle,
    )
    logger.info("Deleted SQS message: receipt=%s...", receipt_handle[:20])
