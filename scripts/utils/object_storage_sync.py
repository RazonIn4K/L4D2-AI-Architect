#!/usr/bin/env python3
"""
Object Storage Sync Utility

Uploads training artifacts to Vultr Object Storage (S3-compatible) or local backup.
Ensures all models, datasets, embeddings, and logs are safely exported before
instance deletion.

Configuration via environment variables:
    VULTR_S3_ENDPOINT - S3 endpoint URL (e.g., sjc1.vultrobjects.com)
    VULTR_S3_ACCESS_KEY - Access key ID
    VULTR_S3_SECRET_KEY - Secret access key
    VULTR_S3_BUCKET - Bucket name

Usage:
    python object_storage_sync.py --sync-all
    python object_storage_sync.py --upload-model model_adapters/l4d2-mistral-v12-lora
    python object_storage_sync.py --upload-dataset data/processed/combined_train.jsonl
    python object_storage_sync.py --upload-embeddings
    python object_storage_sync.py --upload-logs
    python object_storage_sync.py --upload-snapshot snapshot-2024-01-08
    python object_storage_sync.py --local-backup /backup/path
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent))
from security import safe_path, safe_write_json, safe_read_json

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Try to import boto3 for S3 operations
try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    from tqdm.utils import CallbackIOWrapper
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ProgressCallback:
    """Callback for tracking upload progress."""

    def __init__(self, filename: str, filesize: int, desc: str = "Uploading"):
        self.filename = filename
        self.filesize = filesize
        self.desc = desc
        self._seen_so_far = 0

        if HAS_TQDM:
            self.pbar = tqdm(
                total=filesize,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"{desc}: {filename}"
            )
        else:
            self.pbar = None
            print(f"{desc}: {filename} ({filesize / 1024 / 1024:.2f} MB)")

    def __call__(self, bytes_amount: int):
        self._seen_so_far += bytes_amount
        if self.pbar:
            self.pbar.update(bytes_amount)
        elif self._seen_so_far >= self.filesize:
            print(f"  Completed: {self.filename}")

    def close(self):
        if self.pbar:
            self.pbar.close()


class ObjectStorageSync:
    """
    Synchronizes training artifacts to Vultr Object Storage or local backup.

    Supports uploading:
    - LoRA adapters and GGUF models
    - JSONL training datasets
    - Embedding files and FAISS indices
    - TensorBoard training logs
    - Compressed snapshots
    """

    # S3 path prefixes
    PREFIX_MODELS = "models"
    PREFIX_DATASETS = "datasets"
    PREFIX_EMBEDDINGS = "embeddings"
    PREFIX_LOGS = "logs"
    PREFIX_SNAPSHOTS = "snapshots"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        local_backup_path: Optional[str] = None,
    ):
        """
        Initialize the sync client.

        Args:
            endpoint: S3 endpoint URL (or VULTR_S3_ENDPOINT env var)
            access_key: Access key ID (or VULTR_S3_ACCESS_KEY env var)
            secret_key: Secret access key (or VULTR_S3_SECRET_KEY env var)
            bucket: Bucket name (or VULTR_S3_BUCKET env var)
            local_backup_path: Path for local backup (alternative to S3)
        """
        self.endpoint = endpoint or os.environ.get("VULTR_S3_ENDPOINT")
        self.access_key = access_key or os.environ.get("VULTR_S3_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("VULTR_S3_SECRET_KEY")
        self.bucket = bucket or os.environ.get("VULTR_S3_BUCKET")
        self.local_backup_path = local_backup_path

        self.s3_client = None
        self.use_s3 = False

        # Initialize S3 client if credentials available
        if self.endpoint and self.access_key and self.secret_key and self.bucket:
            if not HAS_BOTO3:
                print("Warning: boto3 not installed. Install with: pip install boto3")
            else:
                self._init_s3_client()
        elif not self.local_backup_path:
            print("Warning: No S3 credentials or local backup path configured.")
            print("Set environment variables or use --local-backup")

        # Track uploaded files for manifest
        self.uploaded_files: List[Dict[str, Any]] = []

    def _init_s3_client(self):
        """Initialize the S3 client with Vultr-compatible settings."""
        try:
            # Ensure endpoint has https://
            endpoint_url = self.endpoint
            if not endpoint_url.startswith("http"):
                endpoint_url = f"https://{endpoint_url}"

            self.s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"}
                )
            )

            # Test connection by listing buckets
            self.s3_client.head_bucket(Bucket=self.bucket)
            self.use_s3 = True
            print(f"Connected to S3: {endpoint_url}/{self.bucket}")

        except NoCredentialsError:
            print("Error: Invalid S3 credentials")
            self.use_s3 = False
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404":
                print(f"Error: Bucket '{self.bucket}' not found")
            else:
                print(f"Error connecting to S3: {e}")
            self.use_s3 = False
        except Exception as e:
            print(f"Error initializing S3 client: {e}")
            self.use_s3 = False

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _verify_upload(self, s3_key: str, local_checksum: str) -> bool:
        """Verify upload by comparing checksums."""
        if not self.use_s3:
            return True

        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            # Vultr returns ETag as MD5 for non-multipart uploads
            etag = response.get("ETag", "").strip('"')

            # For multipart uploads, ETag format is different
            if "-" in etag:
                print(f"  Note: Multipart upload, skipping checksum verification")
                return True

            if etag.lower() == local_checksum.lower():
                return True
            else:
                print(f"  Warning: Checksum mismatch for {s3_key}")
                print(f"    Local: {local_checksum}")
                print(f"    Remote: {etag}")
                return False

        except ClientError as e:
            print(f"  Warning: Could not verify upload: {e}")
            return False

    def _upload_file_s3(
        self,
        local_path: Path,
        s3_key: str,
        content_type: Optional[str] = None,
    ) -> bool:
        """Upload a single file to S3 with progress tracking."""
        if not self.use_s3:
            print(f"  Skipping S3 upload (not configured): {local_path}")
            return False

        file_size = local_path.stat().st_size
        checksum = self._compute_checksum(local_path)

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        progress = ProgressCallback(local_path.name, file_size)

        try:
            self.s3_client.upload_file(
                str(local_path),
                self.bucket,
                s3_key,
                Callback=progress,
                ExtraArgs=extra_args if extra_args else None,
            )
            progress.close()

            # Verify upload
            if self._verify_upload(s3_key, checksum):
                self.uploaded_files.append({
                    "local_path": str(local_path),
                    "s3_key": s3_key,
                    "size_bytes": file_size,
                    "checksum_md5": checksum,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                })
                return True
            else:
                return False

        except Exception as e:
            progress.close()
            print(f"  Error uploading {local_path}: {e}")
            return False

    def _copy_file_local(
        self,
        local_path: Path,
        dest_key: str,
    ) -> bool:
        """Copy a file to local backup location."""
        if not self.local_backup_path:
            return False

        dest_path = Path(self.local_backup_path) / dest_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        file_size = local_path.stat().st_size
        checksum = self._compute_checksum(local_path)

        progress = ProgressCallback(local_path.name, file_size, desc="Copying")

        try:
            # Copy with progress tracking
            with open(local_path, "rb") as src:
                with open(dest_path, "wb") as dst:
                    while True:
                        chunk = src.read(8192)
                        if not chunk:
                            break
                        dst.write(chunk)
                        progress(len(chunk))

            progress.close()

            self.uploaded_files.append({
                "local_path": str(local_path),
                "backup_path": str(dest_path),
                "size_bytes": file_size,
                "checksum_md5": checksum,
                "copied_at": datetime.now(timezone.utc).isoformat(),
            })
            return True

        except Exception as e:
            progress.close()
            print(f"  Error copying {local_path}: {e}")
            return False

    def _upload_or_copy(
        self,
        local_path: Path,
        dest_key: str,
        content_type: Optional[str] = None,
    ) -> bool:
        """Upload to S3 or copy to local backup."""
        if self.use_s3:
            return self._upload_file_s3(local_path, dest_key, content_type)
        elif self.local_backup_path:
            return self._copy_file_local(local_path, dest_key)
        else:
            print(f"  No destination configured for: {local_path}")
            return False

    def _upload_directory(
        self,
        local_dir: Path,
        s3_prefix: str,
        file_patterns: Optional[List[str]] = None,
    ) -> Tuple[int, int]:
        """
        Upload all files in a directory.

        Args:
            local_dir: Local directory to upload
            s3_prefix: S3 key prefix
            file_patterns: Optional list of file patterns to include

        Returns:
            Tuple of (successful_uploads, failed_uploads)
        """
        if not local_dir.exists():
            print(f"  Directory not found: {local_dir}")
            return 0, 0

        success_count = 0
        fail_count = 0

        for file_path in local_dir.rglob("*"):
            if file_path.is_dir():
                continue

            # Skip __pycache__ and hidden files
            if "__pycache__" in str(file_path) or file_path.name.startswith("."):
                continue

            # Check file patterns if specified
            if file_patterns:
                if not any(file_path.match(pattern) for pattern in file_patterns):
                    continue

            # Build S3 key
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{relative_path}"

            # Determine content type
            content_type = self._get_content_type(file_path)

            if self._upload_or_copy(file_path, s3_key, content_type):
                success_count += 1
            else:
                fail_count += 1

        return success_count, fail_count

    def _get_content_type(self, file_path: Path) -> Optional[str]:
        """Determine content type based on file extension."""
        suffix = file_path.suffix.lower()
        content_types = {
            ".json": "application/json",
            ".jsonl": "application/jsonlines",
            ".yaml": "application/yaml",
            ".yml": "application/yaml",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".py": "text/x-python",
            ".bin": "application/octet-stream",
            ".safetensors": "application/octet-stream",
            ".gguf": "application/octet-stream",
            ".tar.gz": "application/gzip",
            ".tgz": "application/gzip",
            ".faiss": "application/octet-stream",
        }
        return content_types.get(suffix)

    def upload_model(self, path: str, model_name: Optional[str] = None) -> bool:
        """
        Upload a model (LoRA adapter or GGUF file).

        Args:
            path: Path to model directory or GGUF file
            model_name: Optional name for the model (defaults to directory/file name)

        Returns:
            True if upload successful
        """
        # Validate path
        model_path = safe_path(path, PROJECT_ROOT)

        if not model_path.exists():
            print(f"Error: Model path not found: {model_path}")
            return False

        if model_name is None:
            model_name = model_path.name

        print(f"\nUploading model: {model_name}")

        if model_path.is_file():
            # Single file (e.g., GGUF)
            s3_key = f"{self.PREFIX_MODELS}/{model_name}/{model_path.name}"
            return self._upload_or_copy(model_path, s3_key)
        else:
            # Directory (LoRA adapter)
            s3_prefix = f"{self.PREFIX_MODELS}/{model_name}"
            success, failed = self._upload_directory(model_path, s3_prefix)
            print(f"  Uploaded {success} files, {failed} failed")
            return failed == 0

    def upload_dataset(self, path: str, dataset_name: Optional[str] = None) -> bool:
        """
        Upload a training dataset.

        Args:
            path: Path to dataset file or directory
            dataset_name: Optional name for the dataset

        Returns:
            True if upload successful
        """
        # Validate path
        dataset_path = safe_path(path, PROJECT_ROOT)

        if not dataset_path.exists():
            print(f"Error: Dataset path not found: {dataset_path}")
            return False

        if dataset_name is None:
            dataset_name = dataset_path.stem if dataset_path.is_file() else dataset_path.name

        print(f"\nUploading dataset: {dataset_name}")

        if dataset_path.is_file():
            s3_key = f"{self.PREFIX_DATASETS}/{dataset_path.name}"
            return self._upload_or_copy(dataset_path, s3_key, "application/jsonlines")
        else:
            s3_prefix = f"{self.PREFIX_DATASETS}/{dataset_name}"
            success, failed = self._upload_directory(
                dataset_path,
                s3_prefix,
                file_patterns=["*.jsonl", "*.json", "*.txt"]
            )
            print(f"  Uploaded {success} files, {failed} failed")
            return failed == 0

    def upload_embeddings(self, path: Optional[str] = None) -> bool:
        """
        Upload embedding files and FAISS index.

        Args:
            path: Path to embeddings directory (defaults to data/embeddings)

        Returns:
            True if upload successful
        """
        if path:
            embeddings_path = safe_path(path, PROJECT_ROOT)
        else:
            embeddings_path = PROJECT_ROOT / "data" / "embeddings"

        if not embeddings_path.exists():
            print(f"Embeddings directory not found: {embeddings_path}")
            return False

        print(f"\nUploading embeddings from: {embeddings_path}")

        success, failed = self._upload_directory(
            embeddings_path,
            self.PREFIX_EMBEDDINGS,
            file_patterns=["*.faiss", "*.npy", "*.pkl", "*.json", "*.bin"]
        )
        print(f"  Uploaded {success} files, {failed} failed")
        return failed == 0

    def upload_logs(self, path: Optional[str] = None) -> bool:
        """
        Upload training logs.

        Args:
            path: Path to logs directory (defaults to data/training_logs)

        Returns:
            True if upload successful
        """
        if path:
            logs_path = safe_path(path, PROJECT_ROOT)
        else:
            logs_path = PROJECT_ROOT / "data" / "training_logs"

        if not logs_path.exists():
            print(f"Logs directory not found: {logs_path}")
            return False

        print(f"\nUploading training logs from: {logs_path}")

        # Create timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_prefix = f"{self.PREFIX_LOGS}/{timestamp}"

        success, failed = self._upload_directory(logs_path, s3_prefix)
        print(f"  Uploaded {success} files, {failed} failed")
        return failed == 0

    def upload_snapshot(
        self,
        path: str,
        snapshot_name: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
    ) -> bool:
        """
        Create and upload a compressed snapshot.

        Args:
            path: Path to directory to snapshot
            snapshot_name: Name for the snapshot
            include_patterns: Optional patterns for files to include

        Returns:
            True if upload successful
        """
        # Validate path
        source_path = safe_path(path, PROJECT_ROOT)

        if not source_path.exists():
            print(f"Error: Source path not found: {source_path}")
            return False

        if snapshot_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"{source_path.name}_{timestamp}"

        print(f"\nCreating snapshot: {snapshot_name}")

        # Create temporary tarball
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Create tarball
            print(f"  Compressing {source_path}...")
            with tarfile.open(tmp_path, "w:gz") as tar:
                if source_path.is_file():
                    tar.add(source_path, arcname=source_path.name)
                else:
                    for item in source_path.rglob("*"):
                        if item.is_file():
                            # Skip unwanted files
                            if "__pycache__" in str(item) or item.name.startswith("."):
                                continue
                            if include_patterns:
                                if not any(item.match(p) for p in include_patterns):
                                    continue
                            arcname = item.relative_to(source_path.parent)
                            tar.add(item, arcname=arcname)

            # Upload tarball
            s3_key = f"{self.PREFIX_SNAPSHOTS}/{snapshot_name}.tar.gz"
            result = self._upload_or_copy(tmp_path, s3_key, "application/gzip")

            return result

        finally:
            # Cleanup temp file
            if tmp_path.exists():
                tmp_path.unlink()

    def sync_all(self) -> Dict[str, Any]:
        """
        Sync all artifacts to storage.

        Uploads:
        - model_adapters/ -> /models
        - data/processed/ -> /datasets
        - data/embeddings/ -> /embeddings
        - data/training_logs/ -> /logs

        Returns:
            Summary of sync operation
        """
        print("\n" + "=" * 60)
        print("Starting full sync to object storage")
        print("=" * 60)

        results = {
            "models": {"success": 0, "failed": 0},
            "datasets": {"success": 0, "failed": 0},
            "embeddings": {"success": 0, "failed": 0},
            "logs": {"success": 0, "failed": 0},
        }

        # Sync models
        models_dir = PROJECT_ROOT / "model_adapters"
        if models_dir.exists():
            print(f"\n--- Syncing Models ---")
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith("."):
                    if self.upload_model(str(model_dir)):
                        results["models"]["success"] += 1
                    else:
                        results["models"]["failed"] += 1
        else:
            print(f"\nModels directory not found: {models_dir}")

        # Sync datasets
        datasets_dir = PROJECT_ROOT / "data" / "processed"
        if datasets_dir.exists():
            print(f"\n--- Syncing Datasets ---")
            for dataset_file in datasets_dir.glob("*.jsonl"):
                if self.upload_dataset(str(dataset_file)):
                    results["datasets"]["success"] += 1
                else:
                    results["datasets"]["failed"] += 1
        else:
            print(f"\nDatasets directory not found: {datasets_dir}")

        # Sync embeddings
        embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
        if embeddings_dir.exists():
            print(f"\n--- Syncing Embeddings ---")
            if self.upload_embeddings():
                results["embeddings"]["success"] = 1
            else:
                results["embeddings"]["failed"] = 1
        else:
            print(f"\nEmbeddings directory not found: {embeddings_dir}")

        # Sync logs
        logs_dir = PROJECT_ROOT / "data" / "training_logs"
        if logs_dir.exists():
            print(f"\n--- Syncing Training Logs ---")
            if self.upload_logs():
                results["logs"]["success"] = 1
            else:
                results["logs"]["failed"] = 1
        else:
            print(f"\nLogs directory not found: {logs_dir}")

        # Generate manifest
        self._save_manifest()

        # Print summary
        print("\n" + "=" * 60)
        print("Sync Summary")
        print("=" * 60)
        for category, counts in results.items():
            print(f"  {category}: {counts['success']} success, {counts['failed']} failed")
        print(f"  Total files uploaded: {len(self.uploaded_files)}")

        return results

    def _save_manifest(self):
        """Save manifest of all uploaded files."""
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "bucket": self.bucket if self.use_s3 else None,
            "endpoint": self.endpoint if self.use_s3 else None,
            "local_backup": self.local_backup_path,
            "total_files": len(self.uploaded_files),
            "total_size_bytes": sum(f.get("size_bytes", 0) for f in self.uploaded_files),
            "files": self.uploaded_files,
        }

        # Save manifest to S3/local
        manifest_key = "manifest.json"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8"
        ) as f:
            json.dump(manifest, f, indent=2)
            tmp_path = Path(f.name)

        try:
            self._upload_or_copy(tmp_path, manifest_key, "application/json")
            print(f"\nManifest saved: {manifest_key}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # Also save locally
        local_manifest = PROJECT_ROOT / "data" / "sync_manifest.json"
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        safe_write_json(str(local_manifest), manifest, PROJECT_ROOT)
        print(f"Local manifest saved: {local_manifest}")

    def list_remote_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in the remote bucket."""
        if not self.use_s3:
            print("S3 not configured")
            return []

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )

            files = []
            for obj in response.get("Contents", []):
                files.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                })

            return files

        except ClientError as e:
            print(f"Error listing files: {e}")
            return []


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sync training artifacts to Vultr Object Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync all artifacts
  python object_storage_sync.py --sync-all

  # Upload a specific model
  python object_storage_sync.py --upload-model model_adapters/l4d2-mistral-v12-lora

  # Upload a dataset
  python object_storage_sync.py --upload-dataset data/processed/combined_train.jsonl

  # Upload embeddings
  python object_storage_sync.py --upload-embeddings

  # Upload logs
  python object_storage_sync.py --upload-logs

  # Create and upload a snapshot
  python object_storage_sync.py --upload-snapshot model_adapters --snapshot-name backup-v12

  # Use local backup instead of S3
  python object_storage_sync.py --sync-all --local-backup /backup/l4d2

  # List remote files
  python object_storage_sync.py --list-remote

Environment Variables:
  VULTR_S3_ENDPOINT    - S3 endpoint (e.g., sjc1.vultrobjects.com)
  VULTR_S3_ACCESS_KEY  - Access key ID
  VULTR_S3_SECRET_KEY  - Secret access key
  VULTR_S3_BUCKET      - Bucket name
        """
    )

    # Operation modes
    parser.add_argument(
        "--sync-all",
        action="store_true",
        help="Sync all artifacts (models, datasets, embeddings, logs)"
    )
    parser.add_argument(
        "--upload-model",
        metavar="PATH",
        help="Upload a model (LoRA adapter or GGUF file)"
    )
    parser.add_argument(
        "--model-name",
        help="Custom name for the uploaded model"
    )
    parser.add_argument(
        "--upload-dataset",
        metavar="PATH",
        help="Upload a dataset file or directory"
    )
    parser.add_argument(
        "--dataset-name",
        help="Custom name for the uploaded dataset"
    )
    parser.add_argument(
        "--upload-embeddings",
        action="store_true",
        help="Upload embeddings directory"
    )
    parser.add_argument(
        "--embeddings-path",
        help="Custom path to embeddings directory"
    )
    parser.add_argument(
        "--upload-logs",
        action="store_true",
        help="Upload training logs"
    )
    parser.add_argument(
        "--logs-path",
        help="Custom path to logs directory"
    )
    parser.add_argument(
        "--upload-snapshot",
        metavar="PATH",
        help="Create and upload a compressed snapshot"
    )
    parser.add_argument(
        "--snapshot-name",
        help="Custom name for the snapshot"
    )
    parser.add_argument(
        "--list-remote",
        action="store_true",
        help="List files in remote bucket"
    )
    parser.add_argument(
        "--list-prefix",
        default="",
        help="Prefix filter for listing remote files"
    )

    # Configuration
    parser.add_argument(
        "--local-backup",
        metavar="PATH",
        help="Use local backup instead of S3"
    )
    parser.add_argument(
        "--endpoint",
        help="Override VULTR_S3_ENDPOINT"
    )
    parser.add_argument(
        "--bucket",
        help="Override VULTR_S3_BUCKET"
    )

    args = parser.parse_args()

    # Require at least one operation
    operations = [
        args.sync_all,
        args.upload_model,
        args.upload_dataset,
        args.upload_embeddings,
        args.upload_logs,
        args.upload_snapshot,
        args.list_remote,
    ]

    if not any(operations):
        parser.print_help()
        sys.exit(1)

    # Initialize sync client
    sync = ObjectStorageSync(
        endpoint=args.endpoint,
        bucket=args.bucket,
        local_backup_path=args.local_backup,
    )

    # Check if we have a valid destination
    if not sync.use_s3 and not sync.local_backup_path:
        print("\nError: No valid destination configured.")
        print("Either set S3 environment variables or use --local-backup")
        sys.exit(1)

    # Execute requested operations
    success = True

    if args.list_remote:
        files = sync.list_remote_files(args.list_prefix)
        if files:
            print(f"\nFiles in bucket (prefix: '{args.list_prefix}'):")
            for f in files:
                size_mb = f["size"] / 1024 / 1024
                print(f"  {f['key']} ({size_mb:.2f} MB)")
        else:
            print("No files found")

    if args.sync_all:
        results = sync.sync_all()
        total_failed = sum(r["failed"] for r in results.values())
        success = total_failed == 0

    if args.upload_model:
        success = sync.upload_model(args.upload_model, args.model_name) and success

    if args.upload_dataset:
        success = sync.upload_dataset(args.upload_dataset, args.dataset_name) and success

    if args.upload_embeddings:
        success = sync.upload_embeddings(args.embeddings_path) and success

    if args.upload_logs:
        success = sync.upload_logs(args.logs_path) and success

    if args.upload_snapshot:
        success = sync.upload_snapshot(
            args.upload_snapshot,
            args.snapshot_name
        ) and success

    # Save manifest if any files were uploaded (excluding sync_all which saves its own)
    if not args.sync_all and sync.uploaded_files:
        sync._save_manifest()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
