#!/usr/bin/env python3
"""
Training Job Scheduler for L4D2-AI-Architect

A comprehensive job scheduler for managing ML training pipelines with support for:
- Multiple job types (LLM, RL, embeddings, benchmarks, GGUF export)
- Job dependencies and priority scheduling
- Resource-aware scheduling (GPU memory, CPU)
- Delayed/scheduled execution
- Email and webhook notifications
- Auto-retry on failure
- Persistent job queue

Usage:
    # Add jobs to queue
    python training_scheduler.py add llm --config v14 --priority high
    python training_scheduler.py add rl --personality aggressive --timesteps 500000
    python training_scheduler.py add embedding
    python training_scheduler.py add benchmark --model ollama
    python training_scheduler.py add export --adapter l4d2-mistral-v14-lora/final

    # Manage jobs
    python training_scheduler.py list
    python training_scheduler.py list --status pending
    python training_scheduler.py status <job_id>
    python training_scheduler.py cancel <job_id>
    python training_scheduler.py retry <job_id>

    # Run scheduler
    python training_scheduler.py run
    python training_scheduler.py run --once
    python training_scheduler.py run --dry-run

    # Dependencies
    python training_scheduler.py add llm --config v14 --name train-v14
    python training_scheduler.py add export --adapter v14 --depends-on train-v14

    # Delayed execution
    python training_scheduler.py add llm --config v14 --delay 3600  # 1 hour delay
    python training_scheduler.py add benchmark --start-at "2024-01-15 22:00"
"""

import argparse
import json
import logging
import os
import signal
import smtplib
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_read_json, safe_write_json, validate_url

# Allowed webhook domains for SSRF prevention
WEBHOOK_ALLOWED_DOMAINS = {
    "hooks.slack.com",
    "discord.com",
    "discordapp.com",
    "api.telegram.org",
    "api.pushover.net",
    "ntfy.sh",
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path(__file__).parent.parent.parent / "data" / "scheduler" / "scheduler.log"
        )
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCHEDULER_DIR = PROJECT_ROOT / "data" / "scheduler"
JOBS_FILE = SCHEDULER_DIR / "jobs.json"
CONFIG_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class JobType(str, Enum):
    """Supported job types."""
    LLM = "llm"              # LLM fine-tuning
    RL = "rl"                # RL training
    EMBEDDING = "embedding"  # Embedding generation
    BENCHMARK = "benchmark"  # Benchmark evaluation
    EXPORT = "export"        # GGUF export


class JobStatus(str, Enum):
    """Job status values."""
    PENDING = "pending"
    SCHEDULED = "scheduled"   # Waiting for start time
    WAITING = "waiting"       # Waiting for dependencies
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class Priority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other):
        order = [Priority.LOW, Priority.NORMAL, Priority.HIGH, Priority.CRITICAL]
        return order.index(self) < order.index(other)


@dataclass
class ResourceRequirements:
    """Resource requirements for a job."""
    gpu_memory_gb: float = 0.0    # Required GPU memory in GB
    cpu_cores: int = 1            # Required CPU cores
    ram_gb: float = 4.0           # Required RAM in GB
    estimated_duration_min: int = 60  # Estimated duration in minutes


@dataclass
class NotificationConfig:
    """Notification configuration."""
    email: Optional[str] = None
    webhook_url: Optional[str] = None
    notify_on_start: bool = False
    notify_on_complete: bool = True
    notify_on_failure: bool = True


@dataclass
class Job:
    """Represents a training job."""
    id: str
    name: str
    job_type: JobType
    status: JobStatus
    priority: Priority
    config: Dict[str, Any]
    resources: ResourceRequirements
    notifications: NotificationConfig

    # Timing
    created_at: str
    scheduled_start: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay_seconds: int = 300

    # Results
    output_dir: Optional[str] = None
    error_message: Optional[str] = None
    exit_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['job_type'] = self.job_type.value
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Create Job from dictionary."""
        data['job_type'] = JobType(data['job_type'])
        data['status'] = JobStatus(data['status'])
        data['priority'] = Priority(data['priority'])
        data['resources'] = ResourceRequirements(**data['resources'])
        data['notifications'] = NotificationConfig(**data['notifications'])
        return cls(**data)


# =============================================================================
# RESOURCE MONITOR
# =============================================================================

class ResourceMonitor:
    """Monitors system resources for scheduling decisions."""

    def __init__(self):
        self.last_check = None
        self.cached_resources = {}

    def get_gpu_memory(self) -> Tuple[float, float]:
        """Get GPU memory (total, available) in GB."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                line = result.stdout.strip().split('\n')[0]
                total, free = map(float, line.split(','))
                return total / 1024, free / 1024  # Convert MB to GB
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        return 0.0, 0.0

    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0

    def get_ram_available(self) -> float:
        """Get available RAM in GB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            return 16.0  # Assume 16GB available if psutil not installed

    def can_run_job(self, job: Job) -> Tuple[bool, str]:
        """Check if resources are available for a job."""
        # Check GPU memory
        if job.resources.gpu_memory_gb > 0:
            total_gpu, free_gpu = self.get_gpu_memory()
            if free_gpu < job.resources.gpu_memory_gb:
                return False, f"Insufficient GPU memory: {free_gpu:.1f}GB free, {job.resources.gpu_memory_gb:.1f}GB required"

        # Check RAM
        ram_available = self.get_ram_available()
        if ram_available < job.resources.ram_gb:
            return False, f"Insufficient RAM: {ram_available:.1f}GB free, {job.resources.ram_gb:.1f}GB required"

        return True, "Resources available"


# =============================================================================
# NOTIFICATION SERVICE
# =============================================================================

class NotificationService:
    """Handles job notifications via email and webhooks."""

    def __init__(self, smtp_config: Optional[Dict[str, str]] = None):
        self.smtp_config = smtp_config or {}

    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email notification."""
        if not self.smtp_config:
            logger.warning("SMTP not configured, skipping email notification")
            return False

        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.smtp_config.get('from', 'scheduler@l4d2-ai.local')
            msg['To'] = to

            with smtplib.SMTP(
                self.smtp_config.get('host', 'localhost'),
                int(self.smtp_config.get('port', 25))
            ) as server:
                if self.smtp_config.get('use_tls'):
                    server.starttls()
                if self.smtp_config.get('username'):
                    server.login(
                        self.smtp_config['username'],
                        self.smtp_config.get('password', '')
                    )
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
        """Send webhook notification.

        Args:
            url: Webhook URL (must be HTTPS and from an allowed domain)
            payload: JSON payload to send

        Returns:
            True if webhook was sent successfully, False otherwise
        """
        try:
            import urllib.request
            import urllib.error

            # Validate URL to prevent SSRF - only allow HTTPS and trusted webhook domains
            validated_url = validate_url(
                url,
                allowed_domains=WEBHOOK_ALLOWED_DOMAINS,
                allowed_schemes={"https"}
            )

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                validated_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status == 200
        except ValueError as e:
            logger.error(f"Invalid webhook URL: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False

    def notify_job_event(self, job: Job, event: str, details: str = ""):
        """Send notifications for job event."""
        notif = job.notifications

        # Check if we should notify for this event
        if event == "started" and not notif.notify_on_start:
            return
        if event == "completed" and not notif.notify_on_complete:
            return
        if event == "failed" and not notif.notify_on_failure:
            return

        subject = f"[L4D2 Scheduler] Job {job.name} {event}"
        body = f"""
Job: {job.name} ({job.id})
Type: {job.job_type.value}
Status: {job.status.value}
Event: {event}
Time: {datetime.now().isoformat()}

{details}

Config: {json.dumps(job.config, indent=2)}
"""

        if notif.email:
            self.send_email(notif.email, subject, body)

        if notif.webhook_url:
            self.send_webhook(notif.webhook_url, {
                'job_id': job.id,
                'job_name': job.name,
                'event': event,
                'status': job.status.value,
                'timestamp': datetime.now().isoformat(),
                'details': details
            })


# =============================================================================
# JOB EXECUTOR
# =============================================================================

class JobExecutor:
    """Executes training jobs."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.current_process: Optional[subprocess.Popen] = None

    def execute(self, job: Job) -> Tuple[int, str, str]:
        """
        Execute a job and return (exit_code, stdout, stderr).
        """
        cmd = self._build_command(job)
        logger.info(f"Executing job {job.id}: {' '.join(cmd)}")

        try:
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_root),
                text=True
            )

            stdout, stderr = self.current_process.communicate()
            exit_code = self.current_process.returncode
            self.current_process = None

            return exit_code, stdout, stderr

        except Exception as e:
            logger.error(f"Job execution error: {e}")
            return 1, "", str(e)

    def cancel_current(self):
        """Cancel currently running job."""
        if self.current_process:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.current_process.kill()

    def _build_command(self, job: Job) -> List[str]:
        """Build command for job execution."""
        python = sys.executable
        config = job.config

        if job.job_type == JobType.LLM:
            cmd = [python, str(SCRIPTS_DIR / "training" / "train_unsloth.py")]
            if config.get('config_file'):
                cmd.extend(['--config', str(CONFIG_DIR / config['config_file'])])
            if config.get('model'):
                cmd.extend(['--model', config['model']])
            if config.get('dataset'):
                cmd.extend(['--dataset', config['dataset']])
            if config.get('epochs'):
                cmd.extend(['--epochs', str(config['epochs'])])
            if config.get('batch_size'):
                cmd.extend(['--batch-size', str(config['batch_size'])])
            return cmd

        elif job.job_type == JobType.RL:
            cmd = [python, str(SCRIPTS_DIR / "rl_training" / "train_ppo.py")]
            if config.get('timesteps'):
                cmd.extend(['--timesteps', str(config['timesteps'])])
            if config.get('personality'):
                cmd.extend(['--personality', config['personality']])
            if config.get('save_path'):
                cmd.extend(['--save-path', config['save_path']])
            return cmd

        elif job.job_type == JobType.EMBEDDING:
            cmd = [python, str(SCRIPTS_DIR / "training" / "generate_embeddings.py")]
            if config.get('model'):
                cmd.extend(['--model', config['model']])
            if config.get('input_file'):
                cmd.extend(['--input', config['input_file']])
            return cmd

        elif job.job_type == JobType.BENCHMARK:
            cmd = [python, str(SCRIPTS_DIR / "evaluation" / "benchmark_suite.py")]
            if config.get('model_type'):
                cmd.extend(['--model', config['model_type']])
            if config.get('model_id'):
                cmd.extend(['--model-id', config['model_id']])
            if config.get('output'):
                cmd.extend(['--output', config['output']])
            if config.get('quick'):
                cmd.append('--quick')
            return cmd

        elif job.job_type == JobType.EXPORT:
            cmd = [python, str(SCRIPTS_DIR / "training" / "export_gguf_cpu.py")]
            if config.get('adapter'):
                cmd.extend(['--adapter', config['adapter']])
            if config.get('output'):
                cmd.extend(['--output', config['output']])
            if config.get('quantization'):
                cmd.extend(['--quant', config['quantization']])
            return cmd

        raise ValueError(f"Unknown job type: {job.job_type}")


# =============================================================================
# JOB QUEUE
# =============================================================================

class JobQueue:
    """Manages the job queue with persistence."""

    def __init__(self, jobs_file: Path):
        self.jobs_file = jobs_file
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load jobs from file."""
        if self.jobs_file.exists():
            try:
                data = safe_read_json(str(self.jobs_file), PROJECT_ROOT)
                self.jobs = {
                    job_id: Job.from_dict(job_data)
                    for job_id, job_data in data.get('jobs', {}).items()
                }
                logger.info(f"Loaded {len(self.jobs)} jobs from queue")
            except Exception as e:
                logger.error(f"Failed to load jobs: {e}")
                self.jobs = {}
        else:
            self.jobs = {}

    def _save(self):
        """Save jobs to file."""
        with self._lock:
            data = {
                'version': '1.0',
                'updated_at': datetime.now().isoformat(),
                'jobs': {job_id: job.to_dict() for job_id, job in self.jobs.items()}
            }
            # Ensure directory exists
            self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
            safe_write_json(str(self.jobs_file), data, PROJECT_ROOT)

    def add(self, job: Job) -> str:
        """Add a job to the queue."""
        with self._lock:
            self.jobs[job.id] = job
        self._save()
        logger.info(f"Added job {job.id}: {job.name}")
        return job.id

    def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_by_name(self, name: str) -> Optional[Job]:
        """Get a job by name."""
        for job in self.jobs.values():
            if job.name == name:
                return job
        return None

    def update(self, job: Job):
        """Update a job."""
        with self._lock:
            self.jobs[job.id] = job
        self._save()

    def remove(self, job_id: str) -> bool:
        """Remove a job from the queue."""
        with self._lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                self._save()
                return True
        return False

    def get_pending_jobs(self) -> List[Job]:
        """Get all pending jobs sorted by priority."""
        now = datetime.now()
        pending = []

        for job in self.jobs.values():
            if job.status == JobStatus.PENDING:
                pending.append(job)
            elif job.status == JobStatus.SCHEDULED:
                if job.scheduled_start:
                    start_time = datetime.fromisoformat(job.scheduled_start)
                    if now >= start_time:
                        pending.append(job)

        # Sort by priority (highest first), then by creation time
        pending.sort(key=lambda j: (
            -[Priority.LOW, Priority.NORMAL, Priority.HIGH, Priority.CRITICAL].index(j.priority),
            j.created_at
        ))

        return pending

    def get_jobs_by_status(self, status: JobStatus) -> List[Job]:
        """Get all jobs with a specific status."""
        return [j for j in self.jobs.values() if j.status == status]

    def check_dependencies(self, job: Job) -> Tuple[bool, str]:
        """Check if all dependencies are satisfied."""
        if not job.depends_on:
            return True, "No dependencies"

        for dep_name in job.depends_on:
            dep_job = self.get_by_name(dep_name)
            if not dep_job:
                # Also try by ID
                dep_job = self.get(dep_name)

            if not dep_job:
                return False, f"Dependency not found: {dep_name}"

            if dep_job.status == JobStatus.FAILED:
                return False, f"Dependency failed: {dep_name}"

            if dep_job.status != JobStatus.COMPLETED:
                return False, f"Waiting for dependency: {dep_name}"

        return True, "All dependencies satisfied"

    def list_all(self) -> List[Job]:
        """List all jobs."""
        return list(self.jobs.values())


# =============================================================================
# SCHEDULER
# =============================================================================

class TrainingScheduler:
    """Main scheduler that manages job execution."""

    def __init__(
        self,
        jobs_file: Path = JOBS_FILE,
        check_interval: int = 30,
        smtp_config: Optional[Dict[str, str]] = None
    ):
        self.queue = JobQueue(jobs_file)
        self.executor = JobExecutor(PROJECT_ROOT)
        self.resource_monitor = ResourceMonitor()
        self.notification_service = NotificationService(smtp_config)
        self.check_interval = check_interval
        self.running = False
        self._stop_event = threading.Event()

    def add_job(
        self,
        job_type: JobType,
        name: Optional[str] = None,
        priority: Priority = Priority.NORMAL,
        config: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        scheduled_start: Optional[datetime] = None,
        delay_seconds: Optional[int] = None,
        max_retries: int = 3,
        notifications: Optional[NotificationConfig] = None
    ) -> str:
        """Add a new job to the queue."""
        job_id = str(uuid.uuid4())[:8]

        if name is None:
            name = f"{job_type.value}-{job_id}"

        # Calculate scheduled start time
        start_time = None
        status = JobStatus.PENDING

        if scheduled_start:
            start_time = scheduled_start.isoformat()
            status = JobStatus.SCHEDULED
        elif delay_seconds:
            start_time = (datetime.now() + timedelta(seconds=delay_seconds)).isoformat()
            status = JobStatus.SCHEDULED

        # Set resource requirements based on job type
        resources = self._default_resources(job_type)

        job = Job(
            id=job_id,
            name=name,
            job_type=job_type,
            status=status,
            priority=priority,
            config=config or {},
            resources=resources,
            notifications=notifications or NotificationConfig(),
            created_at=datetime.now().isoformat(),
            scheduled_start=start_time,
            depends_on=depends_on or [],
            max_retries=max_retries
        )

        return self.queue.add(job)

    def _default_resources(self, job_type: JobType) -> ResourceRequirements:
        """Get default resource requirements for job type."""
        defaults = {
            JobType.LLM: ResourceRequirements(gpu_memory_gb=40.0, cpu_cores=4, ram_gb=32.0, estimated_duration_min=120),
            JobType.RL: ResourceRequirements(gpu_memory_gb=8.0, cpu_cores=8, ram_gb=16.0, estimated_duration_min=60),
            JobType.EMBEDDING: ResourceRequirements(gpu_memory_gb=4.0, cpu_cores=2, ram_gb=8.0, estimated_duration_min=30),
            JobType.BENCHMARK: ResourceRequirements(gpu_memory_gb=8.0, cpu_cores=2, ram_gb=8.0, estimated_duration_min=45),
            JobType.EXPORT: ResourceRequirements(gpu_memory_gb=24.0, cpu_cores=4, ram_gb=32.0, estimated_duration_min=30),
        }
        return defaults.get(job_type, ResourceRequirements())

    def run_once(self, dry_run: bool = False) -> Optional[str]:
        """Run a single job if available."""
        pending = self.queue.get_pending_jobs()

        for job in pending:
            # Check dependencies
            deps_ok, deps_msg = self.queue.check_dependencies(job)
            if not deps_ok:
                if job.status != JobStatus.WAITING:
                    job.status = JobStatus.WAITING
                    self.queue.update(job)
                logger.info(f"Job {job.id} waiting: {deps_msg}")
                continue

            # Check resources
            resources_ok, resources_msg = self.resource_monitor.can_run_job(job)
            if not resources_ok:
                logger.info(f"Job {job.id} waiting for resources: {resources_msg}")
                continue

            # Execute job
            if dry_run:
                logger.info(f"[DRY RUN] Would execute job: {job.id} ({job.name})")
                return job.id

            return self._execute_job(job)

        return None

    def _execute_job(self, job: Job) -> str:
        """Execute a single job."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        self.queue.update(job)

        self.notification_service.notify_job_event(job, "started")
        logger.info(f"Starting job {job.id}: {job.name}")

        try:
            exit_code, stdout, stderr = self.executor.execute(job)

            job.exit_code = exit_code
            job.completed_at = datetime.now().isoformat()

            if exit_code == 0:
                job.status = JobStatus.COMPLETED
                self.notification_service.notify_job_event(
                    job, "completed",
                    f"Job completed successfully.\n\nOutput:\n{stdout[-2000:]}"
                )
                logger.info(f"Job {job.id} completed successfully")
            else:
                job.error_message = stderr[-2000:] if stderr else f"Exit code: {exit_code}"

                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.status = JobStatus.RETRYING
                    job.scheduled_start = (
                        datetime.now() + timedelta(seconds=job.retry_delay_seconds)
                    ).isoformat()
                    logger.warning(
                        f"Job {job.id} failed, retry {job.retry_count}/{job.max_retries} "
                        f"in {job.retry_delay_seconds}s"
                    )
                else:
                    job.status = JobStatus.FAILED
                    self.notification_service.notify_job_event(
                        job, "failed",
                        f"Job failed after {job.retry_count} retries.\n\nError:\n{job.error_message}"
                    )
                    logger.error(f"Job {job.id} failed: {job.error_message}")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now().isoformat()
            self.notification_service.notify_job_event(job, "failed", str(e))
            logger.exception(f"Job {job.id} exception: {e}")

        self.queue.update(job)
        return job.id

    def run(self, dry_run: bool = False):
        """Run the scheduler continuously."""
        self.running = True
        logger.info("Scheduler started")

        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self.running and not self._stop_event.is_set():
            try:
                job_id = self.run_once(dry_run=dry_run)
                if job_id:
                    logger.info(f"Executed job: {job_id}")
                else:
                    logger.debug("No jobs ready to run")
            except Exception as e:
                logger.exception(f"Scheduler error: {e}")

            # Wait for next check
            self._stop_event.wait(self.check_interval)

        logger.info("Scheduler stopped")

    def stop(self):
        """Stop the scheduler."""
        self.running = False
        self._stop_event.set()
        self.executor.cancel_current()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.queue.get(job_id)
        if not job:
            return False

        if job.status == JobStatus.RUNNING:
            self.executor.cancel_current()

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now().isoformat()
        self.queue.update(job)
        logger.info(f"Cancelled job: {job_id}")
        return True

    def retry_job(self, job_id: str) -> bool:
        """Retry a failed job."""
        job = self.queue.get(job_id)
        if not job:
            return False

        if job.status not in [JobStatus.FAILED, JobStatus.CANCELLED]:
            logger.warning(f"Cannot retry job {job_id} with status {job.status}")
            return False

        job.status = JobStatus.PENDING
        job.retry_count = 0
        job.error_message = None
        job.exit_code = None
        job.started_at = None
        job.completed_at = None
        job.scheduled_start = None
        self.queue.update(job)
        logger.info(f"Retrying job: {job_id}")
        return True


# =============================================================================
# CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="L4D2-AI-Architect Training Job Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new job')
    add_parser.add_argument('job_type', choices=[t.value for t in JobType],
                           help='Type of job to add')
    add_parser.add_argument('--name', help='Job name (for dependencies)')
    add_parser.add_argument('--priority', choices=[p.value for p in Priority],
                           default='normal', help='Job priority')
    add_parser.add_argument('--depends-on', nargs='+', help='Job names this depends on')
    add_parser.add_argument('--delay', type=int, help='Delay start by N seconds')
    add_parser.add_argument('--start-at', help='Start at specific time (ISO format)')
    add_parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries')
    add_parser.add_argument('--email', help='Email for notifications')
    add_parser.add_argument('--webhook', help='Webhook URL for notifications')

    # LLM-specific options
    add_parser.add_argument('--config', help='Config file (e.g., v14, unsloth_config_v14.yaml)')
    add_parser.add_argument('--model', help='Model name')
    add_parser.add_argument('--dataset', help='Dataset path')
    add_parser.add_argument('--epochs', type=int, help='Number of epochs')
    add_parser.add_argument('--batch-size', type=int, help='Batch size')

    # RL-specific options
    add_parser.add_argument('--timesteps', type=int, help='Training timesteps')
    add_parser.add_argument('--personality', help='Bot personality')

    # Benchmark-specific options
    add_parser.add_argument('--model-type', help='Model type (ollama, openai, base)')
    add_parser.add_argument('--model-id', help='Model ID for testing')
    add_parser.add_argument('--quick', action='store_true', help='Quick benchmark')

    # Export-specific options
    add_parser.add_argument('--adapter', help='Adapter path for export')
    add_parser.add_argument('--output', help='Output path')
    add_parser.add_argument('--quantization', help='Quantization type')

    # List command
    list_parser = subparsers.add_parser('list', help='List jobs')
    list_parser.add_argument('--status', choices=[s.value for s in JobStatus],
                            help='Filter by status')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table',
                            help='Output format')

    # Status command
    status_parser = subparsers.add_parser('status', help='Get job status')
    status_parser.add_argument('job_id', help='Job ID')

    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel a job')
    cancel_parser.add_argument('job_id', help='Job ID')

    # Retry command
    retry_parser = subparsers.add_parser('retry', help='Retry a failed job')
    retry_parser.add_argument('job_id', help='Job ID')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run the scheduler')
    run_parser.add_argument('--once', action='store_true', help='Run only one job')
    run_parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    run_parser.add_argument('--interval', type=int, default=30,
                           help='Check interval in seconds')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Add sample jobs for demonstration')

    return parser


def handle_add(args, scheduler: TrainingScheduler) -> int:
    """Handle add command."""
    job_type = JobType(args.job_type)
    config = {}

    # Build config based on job type
    if job_type == JobType.LLM:
        if args.config:
            # Handle shorthand like "v14" -> "unsloth_config_v14.yaml"
            config_file = args.config
            if not config_file.endswith('.yaml'):
                config_file = f"unsloth_config_{config_file}.yaml"
            config['config_file'] = config_file
        if args.model:
            config['model'] = args.model
        if args.dataset:
            config['dataset'] = args.dataset
        if args.epochs:
            config['epochs'] = args.epochs
        if args.batch_size:
            config['batch_size'] = args.batch_size

    elif job_type == JobType.RL:
        if args.timesteps:
            config['timesteps'] = args.timesteps
        if args.personality:
            config['personality'] = args.personality

    elif job_type == JobType.EMBEDDING:
        if args.model:
            config['model'] = args.model

    elif job_type == JobType.BENCHMARK:
        if args.model_type:
            config['model_type'] = args.model_type
        if args.model_id:
            config['model_id'] = args.model_id
        if args.output:
            config['output'] = args.output
        config['quick'] = args.quick

    elif job_type == JobType.EXPORT:
        if args.adapter:
            config['adapter'] = args.adapter
        if args.output:
            config['output'] = args.output
        if args.quantization:
            config['quantization'] = args.quantization

    # Parse scheduled start
    scheduled_start = None
    if args.start_at:
        scheduled_start = datetime.fromisoformat(args.start_at)

    # Notifications
    notifications = NotificationConfig(
        email=args.email,
        webhook_url=args.webhook,
        notify_on_complete=True,
        notify_on_failure=True
    )

    job_id = scheduler.add_job(
        job_type=job_type,
        name=args.name,
        priority=Priority(args.priority),
        config=config,
        depends_on=args.depends_on,
        scheduled_start=scheduled_start,
        delay_seconds=args.delay,
        max_retries=args.max_retries,
        notifications=notifications
    )

    print(f"Added job: {job_id}")
    return 0


def handle_list(args, scheduler: TrainingScheduler) -> int:
    """Handle list command."""
    jobs = scheduler.queue.list_all()

    if args.status:
        status = JobStatus(args.status)
        jobs = [j for j in jobs if j.status == status]

    # Sort by priority and creation time
    jobs.sort(key=lambda j: (
        -[Priority.LOW, Priority.NORMAL, Priority.HIGH, Priority.CRITICAL].index(j.priority),
        j.created_at
    ))

    if args.format == 'json':
        print(json.dumps([j.to_dict() for j in jobs], indent=2))
        return 0

    # Table format
    if not jobs:
        print("No jobs in queue")
        return 0

    print(f"\n{'ID':<10} {'Name':<25} {'Type':<12} {'Status':<12} {'Priority':<10} {'Created':<20}")
    print("-" * 95)

    for job in jobs:
        created = job.created_at[:19] if job.created_at else ""
        print(f"{job.id:<10} {job.name[:24]:<25} {job.job_type.value:<12} "
              f"{job.status.value:<12} {job.priority.value:<10} {created:<20}")

    print(f"\nTotal: {len(jobs)} jobs")
    return 0


def handle_status(args, scheduler: TrainingScheduler) -> int:
    """Handle status command."""
    job = scheduler.queue.get(args.job_id)
    if not job:
        print(f"Job not found: {args.job_id}")
        return 1

    print(json.dumps(job.to_dict(), indent=2))
    return 0


def handle_cancel(args, scheduler: TrainingScheduler) -> int:
    """Handle cancel command."""
    if scheduler.cancel_job(args.job_id):
        print(f"Cancelled job: {args.job_id}")
        return 0
    print(f"Failed to cancel job: {args.job_id}")
    return 1


def handle_retry(args, scheduler: TrainingScheduler) -> int:
    """Handle retry command."""
    if scheduler.retry_job(args.job_id):
        print(f"Retrying job: {args.job_id}")
        return 0
    print(f"Failed to retry job: {args.job_id}")
    return 1


def handle_run(args, scheduler: TrainingScheduler) -> int:
    """Handle run command."""
    scheduler.check_interval = args.interval

    if args.once:
        job_id = scheduler.run_once(dry_run=args.dry_run)
        if job_id:
            print(f"Executed job: {job_id}")
        else:
            print("No jobs ready to run")
        return 0

    print("Starting scheduler... (Ctrl+C to stop)")
    scheduler.run(dry_run=args.dry_run)
    return 0


def handle_demo(args, scheduler: TrainingScheduler) -> int:
    """Add sample jobs for demonstration."""
    print("Adding sample jobs...")

    # LLM training job
    llm_job_id = scheduler.add_job(
        job_type=JobType.LLM,
        name="train-v14",
        priority=Priority.HIGH,
        config={'config_file': 'unsloth_config_v14.yaml'},
        max_retries=2
    )
    print(f"  Added LLM training job: {llm_job_id}")

    # Export job (depends on training)
    export_job_id = scheduler.add_job(
        job_type=JobType.EXPORT,
        name="export-v14",
        priority=Priority.NORMAL,
        config={
            'adapter': 'model_adapters/l4d2-mistral-v14-lora/final',
            'quantization': 'q4_k_m'
        },
        depends_on=['train-v14']
    )
    print(f"  Added GGUF export job: {export_job_id} (depends on train-v14)")

    # Benchmark job (depends on export)
    benchmark_job_id = scheduler.add_job(
        job_type=JobType.BENCHMARK,
        name="benchmark-v14",
        priority=Priority.NORMAL,
        config={
            'model_type': 'ollama',
            'output': 'data/benchmark_v14.json'
        },
        depends_on=['export-v14']
    )
    print(f"  Added benchmark job: {benchmark_job_id} (depends on export-v14)")

    # RL training job (independent)
    rl_job_id = scheduler.add_job(
        job_type=JobType.RL,
        name="train-aggressive-bot",
        priority=Priority.NORMAL,
        config={
            'timesteps': 500000,
            'personality': 'aggressive'
        }
    )
    print(f"  Added RL training job: {rl_job_id}")

    # Embedding job (delayed start)
    embedding_job_id = scheduler.add_job(
        job_type=JobType.EMBEDDING,
        name="generate-embeddings",
        priority=Priority.LOW,
        delay_seconds=300  # Start in 5 minutes
    )
    print(f"  Added embedding job: {embedding_job_id} (delayed 5 min)")

    print(f"\nAdded {5} sample jobs. Use 'list' to view them.")
    print("Use 'run --dry-run' to test execution order.")
    return 0


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Ensure scheduler directory exists
    SCHEDULER_DIR.mkdir(parents=True, exist_ok=True)

    scheduler = TrainingScheduler()

    handlers = {
        'add': handle_add,
        'list': handle_list,
        'status': handle_status,
        'cancel': handle_cancel,
        'retry': handle_retry,
        'run': handle_run,
        'demo': handle_demo
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args, scheduler)

    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
