#!/usr/bin/env python3
"""
Model Performance Tracker

Tracks inference metrics over time for L4D2 code generation models.
Stores metrics in SQLite database and generates performance reports.

Features:
- Latency tracking (p50, p95, p99)
- Throughput metrics (tokens/sec)
- Memory usage monitoring
- Error rate tracking
- Daily summaries and trend analysis
- Regression detection

Usage:
    # Start tracking with Ollama model
    python performance_tracker.py start --model ollama

    # Generate 7-day performance report
    python performance_tracker.py report --days 7

    # Compare models
    python performance_tracker.py compare --models ollama,openai

    # Run benchmarks and record metrics
    python performance_tracker.py benchmark --model ollama --samples 10

    # Show current status
    python performance_tracker.py status

    # Export metrics to JSON
    python performance_tracker.py export --output metrics.json
"""

import argparse
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_json, safe_read_json

PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class ModelType(str, Enum):
    """Supported model types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    LOCAL = "local"
    VLLM = "vllm"


class MetricType(str, Enum):
    """Types of metrics tracked."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ERROR_RATE = "error_rate"
    TOKEN_COUNT = "token_count"


@dataclass
class InferenceMetric:
    """Single inference metric record."""
    model_name: str
    model_type: str
    timestamp: str
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    input_tokens: int
    memory_mb: float
    success: bool
    error_message: Optional[str] = None
    prompt_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PercentileStats:
    """Percentile statistics for latency."""
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    min_val: float
    max_val: float
    mean: float
    std_dev: float
    count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DailySummary:
    """Daily performance summary."""
    date: str
    model_name: str
    model_type: str
    total_requests: int
    successful_requests: int
    error_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_throughput_tps: float
    total_tokens: int
    avg_memory_mb: float
    peak_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrendAnalysis:
    """Trend analysis result."""
    metric_name: str
    period_days: int
    start_value: float
    end_value: float
    change_percent: float
    trend_direction: str  # "improving", "degrading", "stable"
    is_regression: bool
    regression_threshold: float
    data_points: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceReport:
    """Complete performance report."""
    model_name: str
    model_type: str
    report_period_days: int
    generated_at: str
    summary: Dict[str, Any]
    latency_stats: Dict[str, Any]
    throughput_stats: Dict[str, Any]
    memory_stats: Dict[str, Any]
    error_analysis: Dict[str, Any]
    daily_summaries: List[Dict[str, Any]]
    trends: List[Dict[str, Any]]
    regressions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class MetricsDatabase:
    """SQLite database manager for performance metrics."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS inference_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        model_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        tokens_generated INTEGER NOT NULL,
        tokens_per_second REAL NOT NULL,
        input_tokens INTEGER NOT NULL,
        memory_mb REAL NOT NULL,
        success INTEGER NOT NULL,
        error_message TEXT,
        prompt_hash TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_metrics_model ON inference_metrics(model_name, model_type);
    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON inference_metrics(timestamp);
    CREATE INDEX IF NOT EXISTS idx_metrics_success ON inference_metrics(success);

    CREATE TABLE IF NOT EXISTS daily_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_type TEXT NOT NULL,
        total_requests INTEGER NOT NULL,
        successful_requests INTEGER NOT NULL,
        error_rate REAL NOT NULL,
        avg_latency_ms REAL NOT NULL,
        p50_latency_ms REAL NOT NULL,
        p95_latency_ms REAL NOT NULL,
        p99_latency_ms REAL NOT NULL,
        avg_throughput_tps REAL NOT NULL,
        total_tokens INTEGER NOT NULL,
        avg_memory_mb REAL NOT NULL,
        peak_memory_mb REAL NOT NULL,
        UNIQUE(date, model_name, model_type)
    );

    CREATE INDEX IF NOT EXISTS idx_summaries_date ON daily_summaries(date);
    CREATE INDEX IF NOT EXISTS idx_summaries_model ON daily_summaries(model_name, model_type);

    CREATE TABLE IF NOT EXISTS regression_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        detected_at TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_type TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        baseline_value REAL NOT NULL,
        current_value REAL NOT NULL,
        change_percent REAL NOT NULL,
        threshold_percent REAL NOT NULL,
        acknowledged INTEGER DEFAULT 0
    );
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Uses default if None.
        """
        if db_path is None:
            db_path = PROJECT_ROOT / "data" / "metrics" / "performance.db"

        # Validate path stays within project
        self.db_path = safe_path(str(db_path), PROJECT_ROOT, create_parents=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)

    @contextmanager
    def _get_connection(self):
        """Get database connection as context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def record_metric(self, metric: InferenceMetric):
        """Record a single inference metric."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO inference_metrics
                (model_name, model_type, timestamp, latency_ms, tokens_generated,
                 tokens_per_second, input_tokens, memory_mb, success, error_message, prompt_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.model_name,
                metric.model_type,
                metric.timestamp,
                metric.latency_ms,
                metric.tokens_generated,
                metric.tokens_per_second,
                metric.input_tokens,
                metric.memory_mb,
                1 if metric.success else 0,
                metric.error_message,
                metric.prompt_hash
            ))

    def record_metrics_batch(self, metrics: List[InferenceMetric]):
        """Record multiple metrics in batch."""
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT INTO inference_metrics
                (model_name, model_type, timestamp, latency_ms, tokens_generated,
                 tokens_per_second, input_tokens, memory_mb, success, error_message, prompt_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (m.model_name, m.model_type, m.timestamp, m.latency_ms, m.tokens_generated,
                 m.tokens_per_second, m.input_tokens, m.memory_mb,
                 1 if m.success else 0, m.error_message, m.prompt_hash)
                for m in metrics
            ])

    def get_metrics(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """Query metrics with optional filters."""
        query = "SELECT * FROM inference_metrics WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += f" ORDER BY timestamp DESC LIMIT {int(limit)}"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_latency_percentiles(
        self,
        model_name: str,
        model_type: str,
        days: int = 7
    ) -> Optional[PercentileStats]:
        """Calculate latency percentiles for a model."""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT latency_ms FROM inference_metrics
                WHERE model_name = ? AND model_type = ?
                AND timestamp >= ? AND success = 1
                ORDER BY latency_ms
            """, (model_name, model_type, start_date))

            latencies = [row[0] for row in cursor.fetchall()]

        if not latencies:
            return None

        count = len(latencies)
        return PercentileStats(
            p50=latencies[int(count * 0.50)] if count > 0 else 0,
            p75=latencies[int(count * 0.75)] if count > 0 else 0,
            p90=latencies[int(count * 0.90)] if count > 0 else 0,
            p95=latencies[int(count * 0.95)] if count > 0 else 0,
            p99=latencies[int(count * 0.99)] if count > 0 else 0,
            min_val=min(latencies),
            max_val=max(latencies),
            mean=statistics.mean(latencies),
            std_dev=statistics.stdev(latencies) if count > 1 else 0,
            count=count
        )

    def save_daily_summary(self, summary: DailySummary):
        """Save or update daily summary."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_summaries
                (date, model_name, model_type, total_requests, successful_requests,
                 error_rate, avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms,
                 avg_throughput_tps, total_tokens, avg_memory_mb, peak_memory_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.date, summary.model_name, summary.model_type,
                summary.total_requests, summary.successful_requests,
                summary.error_rate, summary.avg_latency_ms,
                summary.p50_latency_ms, summary.p95_latency_ms, summary.p99_latency_ms,
                summary.avg_throughput_tps, summary.total_tokens,
                summary.avg_memory_mb, summary.peak_memory_mb
            ))

    def get_daily_summaries(
        self,
        model_name: str,
        model_type: str,
        days: int = 30
    ) -> List[DailySummary]:
        """Get daily summaries for a model."""
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM daily_summaries
                WHERE model_name = ? AND model_type = ? AND date >= ?
                ORDER BY date DESC
            """, (model_name, model_type, start_date))

            return [
                DailySummary(**dict(row))
                for row in cursor.fetchall()
            ]

    def record_regression(
        self,
        model_name: str,
        model_type: str,
        metric_name: str,
        baseline: float,
        current: float,
        change_percent: float,
        threshold: float
    ):
        """Record a performance regression alert."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO regression_alerts
                (detected_at, model_name, model_type, metric_name,
                 baseline_value, current_value, change_percent, threshold_percent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(), model_name, model_type, metric_name,
                baseline, current, change_percent, threshold
            ))

    def get_unacknowledged_regressions(
        self,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get unacknowledged regression alerts."""
        query = "SELECT * FROM regression_alerts WHERE acknowledged = 0"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        query += " ORDER BY detected_at DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_model_list(self) -> List[Tuple[str, str]]:
        """Get list of all tracked models."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT model_name, model_type FROM inference_metrics
                ORDER BY model_name
            """)
            return [(row[0], row[1]) for row in cursor.fetchall()]

    def get_stats_summary(
        self,
        model_name: str,
        model_type: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get summary statistics for a model."""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    AVG(latency_ms) as avg_latency,
                    AVG(tokens_per_second) as avg_throughput,
                    SUM(tokens_generated) as total_tokens,
                    AVG(memory_mb) as avg_memory,
                    MAX(memory_mb) as peak_memory
                FROM inference_metrics
                WHERE model_name = ? AND model_type = ? AND timestamp >= ?
            """, (model_name, model_type, start_date))

            row = cursor.fetchone()
            if row and row[0] > 0:
                return {
                    "total_requests": row[0],
                    "successful_requests": row[1],
                    "error_rate": (row[0] - row[1]) / row[0] * 100 if row[0] > 0 else 0,
                    "avg_latency_ms": row[2] or 0,
                    "avg_throughput_tps": row[3] or 0,
                    "total_tokens": row[4] or 0,
                    "avg_memory_mb": row[5] or 0,
                    "peak_memory_mb": row[6] or 0
                }
            return {}


# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================

class PerformanceTracker:
    """Main performance tracking class."""

    # Regression thresholds (percentage change that triggers alert)
    REGRESSION_THRESHOLDS = {
        "latency_p95": 20.0,      # 20% increase in p95 latency
        "latency_p99": 25.0,      # 25% increase in p99 latency
        "throughput": -15.0,      # 15% decrease in throughput
        "error_rate": 5.0,        # 5% absolute increase in error rate
        "memory": 30.0            # 30% increase in memory usage
    }

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize performance tracker."""
        self.db = MetricsDatabase(db_path)
        self._memory_baseline: Optional[float] = None

    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback if psutil not available
            return 0.0

    def record_inference(
        self,
        model_name: str,
        model_type: str,
        latency_ms: float,
        tokens_generated: int,
        input_tokens: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
        prompt_hash: Optional[str] = None
    ):
        """Record a single inference metric."""
        memory_mb = self.get_current_memory_mb()
        tokens_per_second = (tokens_generated / (latency_ms / 1000)) if latency_ms > 0 else 0

        metric = InferenceMetric(
            model_name=model_name,
            model_type=model_type,
            timestamp=datetime.now().isoformat(),
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            input_tokens=input_tokens,
            memory_mb=memory_mb,
            success=success,
            error_message=error_message,
            prompt_hash=prompt_hash
        )

        self.db.record_metric(metric)

    def compute_daily_summary(
        self,
        model_name: str,
        model_type: str,
        date: Optional[str] = None
    ) -> Optional[DailySummary]:
        """Compute and store daily summary for a model."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Get metrics for the day
        start_date = f"{date}T00:00:00"
        end_date = f"{date}T23:59:59"

        metrics = self.db.get_metrics(
            model_name=model_name,
            model_type=model_type,
            start_date=start_date,
            end_date=end_date
        )

        if not metrics:
            return None

        # Calculate statistics
        successful = [m for m in metrics if m["success"]]
        latencies = [m["latency_ms"] for m in successful]
        throughputs = [m["tokens_per_second"] for m in successful]
        memories = [m["memory_mb"] for m in metrics]

        if not latencies:
            latencies = [0]
        if not throughputs:
            throughputs = [0]

        latencies_sorted = sorted(latencies)
        count = len(latencies_sorted)

        summary = DailySummary(
            date=date,
            model_name=model_name,
            model_type=model_type,
            total_requests=len(metrics),
            successful_requests=len(successful),
            error_rate=(len(metrics) - len(successful)) / len(metrics) * 100 if metrics else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=latencies_sorted[int(count * 0.50)] if count > 0 else 0,
            p95_latency_ms=latencies_sorted[int(count * 0.95)] if count > 0 else 0,
            p99_latency_ms=latencies_sorted[int(count * 0.99)] if count > 0 else 0,
            avg_throughput_tps=statistics.mean(throughputs) if throughputs else 0,
            total_tokens=sum(m["tokens_generated"] for m in metrics),
            avg_memory_mb=statistics.mean(memories) if memories else 0,
            peak_memory_mb=max(memories) if memories else 0
        )

        self.db.save_daily_summary(summary)
        return summary

    def analyze_trends(
        self,
        model_name: str,
        model_type: str,
        days: int = 7
    ) -> List[TrendAnalysis]:
        """Analyze performance trends over time."""
        summaries = self.db.get_daily_summaries(model_name, model_type, days)

        if len(summaries) < 2:
            return []

        trends = []
        metrics_to_analyze = [
            ("p95_latency_ms", "latency_p95", True),   # Higher is worse
            ("p99_latency_ms", "latency_p99", True),   # Higher is worse
            ("avg_throughput_tps", "throughput", False),  # Lower is worse
            ("error_rate", "error_rate", True),        # Higher is worse
            ("avg_memory_mb", "memory", True)          # Higher is worse
        ]

        for attr, threshold_key, higher_is_worse in metrics_to_analyze:
            values = [getattr(s, attr) for s in summaries]

            if not values or values[0] == 0:
                continue

            # Compare first (oldest) to last (newest)
            start_val = values[-1]  # Oldest
            end_val = values[0]     # Newest

            if start_val == 0:
                continue

            change_pct = ((end_val - start_val) / start_val) * 100
            threshold = self.REGRESSION_THRESHOLDS.get(threshold_key, 20.0)

            # Determine trend direction
            if abs(change_pct) < 5:
                direction = "stable"
            elif (higher_is_worse and change_pct > 0) or (not higher_is_worse and change_pct < 0):
                direction = "degrading"
            else:
                direction = "improving"

            # Check for regression
            is_regression = False
            if higher_is_worse and change_pct > threshold:
                is_regression = True
            elif not higher_is_worse and change_pct < threshold:
                is_regression = True

            trends.append(TrendAnalysis(
                metric_name=attr,
                period_days=days,
                start_value=start_val,
                end_value=end_val,
                change_percent=change_pct,
                trend_direction=direction,
                is_regression=is_regression,
                regression_threshold=threshold,
                data_points=len(values)
            ))

            # Record regression if detected
            if is_regression:
                self.db.record_regression(
                    model_name=model_name,
                    model_type=model_type,
                    metric_name=attr,
                    baseline=start_val,
                    current=end_val,
                    change_percent=change_pct,
                    threshold=threshold
                )

        return trends

    def generate_report(
        self,
        model_name: str,
        model_type: str,
        days: int = 7
    ) -> PerformanceReport:
        """Generate comprehensive performance report."""
        # Get latency percentiles
        latency_stats = self.db.get_latency_percentiles(model_name, model_type, days)

        # Get summary statistics
        summary = self.db.get_stats_summary(model_name, model_type, days)

        # Get daily summaries
        daily_summaries = self.db.get_daily_summaries(model_name, model_type, days)

        # Analyze trends
        trends = self.analyze_trends(model_name, model_type, days)

        # Get regressions
        regressions = self.db.get_unacknowledged_regressions(model_name)

        # Calculate throughput stats
        metrics = self.db.get_metrics(
            model_name=model_name,
            model_type=model_type,
            start_date=(datetime.now() - timedelta(days=days)).isoformat()
        )

        throughputs = [m["tokens_per_second"] for m in metrics if m["success"]]
        memories = [m["memory_mb"] for m in metrics]

        throughput_stats = {}
        if throughputs:
            throughputs_sorted = sorted(throughputs)
            count = len(throughputs_sorted)
            throughput_stats = {
                "min": min(throughputs),
                "max": max(throughputs),
                "mean": statistics.mean(throughputs),
                "p50": throughputs_sorted[int(count * 0.50)],
                "p95": throughputs_sorted[int(count * 0.95)] if count > 0 else 0,
                "std_dev": statistics.stdev(throughputs) if count > 1 else 0
            }

        memory_stats = {}
        if memories:
            memory_stats = {
                "min": min(memories),
                "max": max(memories),
                "mean": statistics.mean(memories),
                "std_dev": statistics.stdev(memories) if len(memories) > 1 else 0
            }

        # Analyze errors
        errors = [m for m in metrics if not m["success"]]
        error_analysis = {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(metrics) * 100 if metrics else 0,
            "error_messages": {}
        }
        for e in errors:
            msg = e.get("error_message", "Unknown")
            error_analysis["error_messages"][msg] = error_analysis["error_messages"].get(msg, 0) + 1

        return PerformanceReport(
            model_name=model_name,
            model_type=model_type,
            report_period_days=days,
            generated_at=datetime.now().isoformat(),
            summary=summary,
            latency_stats=latency_stats.to_dict() if latency_stats else {},
            throughput_stats=throughput_stats,
            memory_stats=memory_stats,
            error_analysis=error_analysis,
            daily_summaries=[s.to_dict() for s in daily_summaries],
            trends=[t.to_dict() for t in trends],
            regressions=regressions
        )

    def compare_models(
        self,
        models: List[Tuple[str, str]],
        days: int = 7
    ) -> Dict[str, Any]:
        """Compare performance across multiple models."""
        comparisons = {}

        for model_name, model_type in models:
            stats = self.db.get_stats_summary(model_name, model_type, days)
            percentiles = self.db.get_latency_percentiles(model_name, model_type, days)

            comparisons[f"{model_name}_{model_type}"] = {
                "model_name": model_name,
                "model_type": model_type,
                "summary": stats,
                "latency_percentiles": percentiles.to_dict() if percentiles else {}
            }

        # Determine best model for each metric
        rankings = {
            "lowest_latency": None,
            "highest_throughput": None,
            "lowest_error_rate": None,
            "lowest_memory": None
        }

        valid_models = [k for k, v in comparisons.items() if v.get("summary")]

        if valid_models:
            # Lowest latency
            rankings["lowest_latency"] = min(
                valid_models,
                key=lambda k: comparisons[k]["summary"].get("avg_latency_ms", float("inf"))
            )
            # Highest throughput
            rankings["highest_throughput"] = max(
                valid_models,
                key=lambda k: comparisons[k]["summary"].get("avg_throughput_tps", 0)
            )
            # Lowest error rate
            rankings["lowest_error_rate"] = min(
                valid_models,
                key=lambda k: comparisons[k]["summary"].get("error_rate", float("inf"))
            )
            # Lowest memory
            rankings["lowest_memory"] = min(
                valid_models,
                key=lambda k: comparisons[k]["summary"].get("avg_memory_mb", float("inf"))
            )

        return {
            "period_days": days,
            "generated_at": datetime.now().isoformat(),
            "models": comparisons,
            "rankings": rankings
        }


# =============================================================================
# MODEL RUNNERS
# =============================================================================

class OllamaRunner:
    """Runner for Ollama models."""

    def __init__(self, model_name: str = "l4d2-code-v10plus"):
        """Initialize Ollama runner."""
        self.model_name = model_name
        self._check_ollama()

    def _check_ollama(self):
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                print("Warning: Ollama not responding. Make sure it's running.")
        except FileNotFoundError:
            print("Warning: Ollama not installed. Install from https://ollama.ai")
        except subprocess.TimeoutExpired:
            print("Warning: Ollama timeout. Make sure it's running.")

    def run_inference(self, prompt: str, max_tokens: int = 256) -> Tuple[str, float, int]:
        """Run inference and return (response, latency_ms, token_count)."""
        start_time = time.perf_counter()

        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=60
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            response = result.stdout.strip()

            # Estimate token count (rough approximation)
            token_count = len(response.split()) + len(response) // 4

            return response, latency_ms, token_count

        except subprocess.TimeoutExpired:
            return "", 60000, 0
        except Exception as e:
            return "", 0, 0


class OpenAIRunner:
    """Runner for OpenAI models."""

    def __init__(self, model_id: str, api_key: Optional[str] = None):
        """Initialize OpenAI runner."""
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def run_inference(self, prompt: str, max_tokens: int = 256) -> Tuple[str, float, int]:
        """Run inference and return (response, latency_ms, token_count)."""
        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are an expert SourcePawn developer for L4D2."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            content = response.choices[0].message.content or ""
            token_count = response.usage.completion_tokens if response.usage else len(content.split())

            return content, latency_ms, token_count

        except Exception as e:
            return "", 0, 0


# =============================================================================
# CLI INTERFACE
# =============================================================================

def format_report(report: PerformanceReport) -> str:
    """Format report for terminal output."""
    lines = [
        "",
        "=" * 70,
        f"Performance Report: {report.model_name} ({report.model_type})",
        f"Period: {report.report_period_days} days | Generated: {report.generated_at}",
        "=" * 70,
        "",
        "Summary",
        "-" * 40
    ]

    if report.summary:
        lines.extend([
            f"  Total Requests:     {report.summary.get('total_requests', 0):,}",
            f"  Successful:         {report.summary.get('successful_requests', 0):,}",
            f"  Error Rate:         {report.summary.get('error_rate', 0):.2f}%",
            f"  Avg Latency:        {report.summary.get('avg_latency_ms', 0):.2f} ms",
            f"  Avg Throughput:     {report.summary.get('avg_throughput_tps', 0):.2f} tokens/sec",
            f"  Total Tokens:       {report.summary.get('total_tokens', 0):,}",
            f"  Avg Memory:         {report.summary.get('avg_memory_mb', 0):.2f} MB",
            f"  Peak Memory:        {report.summary.get('peak_memory_mb', 0):.2f} MB"
        ])

    if report.latency_stats:
        lines.extend([
            "",
            "Latency Percentiles",
            "-" * 40,
            f"  p50:    {report.latency_stats.get('p50', 0):.2f} ms",
            f"  p75:    {report.latency_stats.get('p75', 0):.2f} ms",
            f"  p90:    {report.latency_stats.get('p90', 0):.2f} ms",
            f"  p95:    {report.latency_stats.get('p95', 0):.2f} ms",
            f"  p99:    {report.latency_stats.get('p99', 0):.2f} ms",
            f"  Min:    {report.latency_stats.get('min_val', 0):.2f} ms",
            f"  Max:    {report.latency_stats.get('max_val', 0):.2f} ms",
            f"  Mean:   {report.latency_stats.get('mean', 0):.2f} ms"
        ])

    if report.trends:
        lines.extend(["", "Trends", "-" * 40])
        for trend in report.trends:
            direction = trend.get("trend_direction", "stable")
            symbol = {"improving": "+", "degrading": "-", "stable": "="}[direction]
            regression = " [REGRESSION]" if trend.get("is_regression") else ""
            lines.append(
                f"  [{symbol}] {trend['metric_name']}: "
                f"{trend['start_value']:.2f} -> {trend['end_value']:.2f} "
                f"({trend['change_percent']:+.1f}%){regression}"
            )

    if report.regressions:
        lines.extend([
            "",
            "Active Regressions",
            "-" * 40
        ])
        for reg in report.regressions:
            lines.append(
                f"  [{reg['detected_at'][:10]}] {reg['metric_name']}: "
                f"{reg['change_percent']:+.1f}% (threshold: {reg['threshold_percent']}%)"
            )

    lines.extend(["", "=" * 70, ""])
    return "\n".join(lines)


def format_comparison(comparison: Dict[str, Any]) -> str:
    """Format model comparison for terminal output."""
    lines = [
        "",
        "=" * 70,
        f"Model Comparison | Period: {comparison['period_days']} days",
        "=" * 70,
        ""
    ]

    for model_key, data in comparison.get("models", {}).items():
        summary = data.get("summary", {})
        percentiles = data.get("latency_percentiles", {})

        lines.extend([
            f"Model: {data['model_name']} ({data['model_type']})",
            "-" * 40
        ])

        if summary:
            lines.extend([
                f"  Requests:     {summary.get('total_requests', 0):,}",
                f"  Error Rate:   {summary.get('error_rate', 0):.2f}%",
                f"  Avg Latency:  {summary.get('avg_latency_ms', 0):.2f} ms",
                f"  Throughput:   {summary.get('avg_throughput_tps', 0):.2f} t/s",
                f"  Memory:       {summary.get('avg_memory_mb', 0):.2f} MB"
            ])

        if percentiles:
            lines.append(
                f"  Latency (p50/p95/p99): "
                f"{percentiles.get('p50', 0):.0f} / "
                f"{percentiles.get('p95', 0):.0f} / "
                f"{percentiles.get('p99', 0):.0f} ms"
            )

        lines.append("")

    rankings = comparison.get("rankings", {})
    if any(rankings.values()):
        lines.extend([
            "Rankings",
            "-" * 40,
            f"  Lowest Latency:     {rankings.get('lowest_latency', 'N/A')}",
            f"  Highest Throughput: {rankings.get('highest_throughput', 'N/A')}",
            f"  Lowest Error Rate:  {rankings.get('lowest_error_rate', 'N/A')}",
            f"  Lowest Memory:      {rankings.get('lowest_memory', 'N/A')}",
            ""
        ])

    lines.extend(["=" * 70, ""])
    return "\n".join(lines)


def run_benchmark(
    tracker: PerformanceTracker,
    model_type: str,
    model_name: str,
    samples: int,
    model_id: Optional[str] = None
) -> List[InferenceMetric]:
    """Run benchmark and record metrics."""
    print(f"Running benchmark with {samples} samples...")

    # Sample prompts for benchmarking
    prompts = [
        "Write a SourcePawn function to heal all survivors",
        "Create a timer that spawns a Tank every 5 minutes",
        "Hook the player death event and print a message",
        "Write a function to teleport a player to another player",
        "Create a menu that lists all connected players",
        "Write code to give all survivors adrenaline",
        "Hook the infected spawn event and count special infected",
        "Create a command to toggle god mode",
        "Write a function to calculate distance between two players",
        "Create a repeating announcement timer"
    ]

    # Initialize runner
    if model_type == ModelType.OLLAMA.value:
        runner = OllamaRunner(model_name)
    elif model_type == ModelType.OPENAI.value:
        if not model_id:
            print("Error: --model-id required for OpenAI models")
            return []
        runner = OpenAIRunner(model_id)
    else:
        print(f"Error: Unsupported model type: {model_type}")
        return []

    metrics = []
    for i in range(samples):
        prompt = prompts[i % len(prompts)]
        print(f"  Sample {i+1}/{samples}: ", end="", flush=True)

        try:
            response, latency_ms, token_count = runner.run_inference(prompt)
            success = len(response) > 10

            tracker.record_inference(
                model_name=model_name,
                model_type=model_type,
                latency_ms=latency_ms,
                tokens_generated=token_count,
                input_tokens=len(prompt.split()),
                success=success,
                error_message=None if success else "Empty or short response"
            )

            print(f"{latency_ms:.0f}ms, {token_count} tokens {'OK' if success else 'FAIL'}")

        except Exception as e:
            tracker.record_inference(
                model_name=model_name,
                model_type=model_type,
                latency_ms=0,
                tokens_generated=0,
                input_tokens=len(prompt.split()),
                success=False,
                error_message=str(e)
            )
            print(f"ERROR: {e}")

        time.sleep(0.5)  # Rate limiting

    return metrics


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Model Performance Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with Ollama model
  python performance_tracker.py benchmark --model ollama --samples 10

  # Generate performance report
  python performance_tracker.py report --model ollama --days 7

  # Compare models
  python performance_tracker.py compare --models ollama,openai

  # Show current status
  python performance_tracker.py status

  # Export metrics
  python performance_tracker.py export --output metrics.json
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark and record metrics")
    bench_parser.add_argument("--model", choices=["ollama", "openai", "local"], default="ollama",
                              help="Model type to benchmark")
    bench_parser.add_argument("--model-name", default="l4d2-code-v10plus",
                              help="Model name (for Ollama)")
    bench_parser.add_argument("--model-id", help="Model ID (for OpenAI fine-tuned models)")
    bench_parser.add_argument("--samples", type=int, default=10,
                              help="Number of samples to run")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate performance report")
    report_parser.add_argument("--model", default="ollama", help="Model type")
    report_parser.add_argument("--model-name", default="l4d2-code-v10plus", help="Model name")
    report_parser.add_argument("--days", type=int, default=7, help="Report period in days")
    report_parser.add_argument("--output", help="Save report to JSON file")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare model performance")
    compare_parser.add_argument("--models", required=True,
                                help="Comma-separated list of model types (e.g., ollama,openai)")
    compare_parser.add_argument("--days", type=int, default=7, help="Comparison period in days")
    compare_parser.add_argument("--output", help="Save comparison to JSON file")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show current tracking status")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export metrics to JSON")
    export_parser.add_argument("--output", required=True, help="Output file path")
    export_parser.add_argument("--model", help="Filter by model type")
    export_parser.add_argument("--days", type=int, default=30, help="Export period in days")

    # Start command (for continuous tracking)
    start_parser = subparsers.add_parser("start", help="Start tracking session (interactive)")
    start_parser.add_argument("--model", choices=["ollama", "openai"], default="ollama",
                              help="Model type to track")
    start_parser.add_argument("--model-name", default="l4d2-code-v10plus", help="Model name")

    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Compute daily summaries")
    summarize_parser.add_argument("--model", default="ollama", help="Model type")
    summarize_parser.add_argument("--model-name", default="l4d2-code-v10plus", help="Model name")
    summarize_parser.add_argument("--date", help="Date to summarize (YYYY-MM-DD), defaults to today")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    tracker = PerformanceTracker()

    if args.command == "benchmark":
        run_benchmark(
            tracker,
            model_type=args.model,
            model_name=args.model_name,
            samples=args.samples,
            model_id=getattr(args, "model_id", None)
        )
        print("\nBenchmark complete. Run 'report' to see results.")

    elif args.command == "report":
        report = tracker.generate_report(
            model_name=args.model_name,
            model_type=args.model,
            days=args.days
        )
        print(format_report(report))

        if args.output:
            output_path = safe_path(args.output, PROJECT_ROOT, create_parents=True)
            safe_write_json(str(output_path), report.to_dict(), PROJECT_ROOT)
            print(f"Report saved to: {output_path}")

    elif args.command == "compare":
        model_types = args.models.split(",")
        models = []
        for mt in model_types:
            mt = mt.strip()
            if mt == "ollama":
                models.append(("l4d2-code-v10plus", "ollama"))
            elif mt == "openai":
                models.append(("ft:gpt-4o-mini", "openai"))
            else:
                models.append((mt, mt))

        comparison = tracker.compare_models(models, args.days)
        print(format_comparison(comparison))

        if args.output:
            output_path = safe_path(args.output, PROJECT_ROOT, create_parents=True)
            safe_write_json(str(output_path), comparison, PROJECT_ROOT)
            print(f"Comparison saved to: {output_path}")

    elif args.command == "status":
        print("\n" + "=" * 50)
        print("Performance Tracker Status")
        print("=" * 50)

        models = tracker.db.get_model_list()
        if models:
            print(f"\nTracked Models: {len(models)}")
            for name, mtype in models:
                stats = tracker.db.get_stats_summary(name, mtype, 7)
                print(f"  - {name} ({mtype}): {stats.get('total_requests', 0):,} requests (7d)")
        else:
            print("\nNo models tracked yet. Run 'benchmark' to start collecting data.")

        regressions = tracker.db.get_unacknowledged_regressions()
        if regressions:
            print(f"\nActive Regressions: {len(regressions)}")
            for reg in regressions[:5]:
                print(f"  - {reg['model_name']}: {reg['metric_name']} ({reg['change_percent']:+.1f}%)")

        print("\n" + "=" * 50 + "\n")

    elif args.command == "export":
        start_date = (datetime.now() - timedelta(days=args.days)).isoformat()
        metrics = tracker.db.get_metrics(
            model_type=args.model,
            start_date=start_date
        )

        output_path = safe_path(args.output, PROJECT_ROOT, create_parents=True)
        safe_write_json(str(output_path), {"metrics": metrics, "exported_at": datetime.now().isoformat()}, PROJECT_ROOT)
        print(f"Exported {len(metrics)} metrics to: {output_path}")

    elif args.command == "start":
        print(f"\nStarting interactive tracking session for {args.model_name} ({args.model})")
        print("Enter prompts to track inference, or 'quit' to exit.\n")

        if args.model == "ollama":
            runner = OllamaRunner(args.model_name)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY environment variable required")
                return
            runner = OpenAIRunner(args.model_name, api_key)

        while True:
            try:
                prompt = input("Prompt> ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue

                print("Running inference...", end=" ", flush=True)
                response, latency_ms, token_count = runner.run_inference(prompt)
                success = len(response) > 10

                tracker.record_inference(
                    model_name=args.model_name,
                    model_type=args.model,
                    latency_ms=latency_ms,
                    tokens_generated=token_count,
                    input_tokens=len(prompt.split()),
                    success=success
                )

                print(f"Done ({latency_ms:.0f}ms, {token_count} tokens)")
                print(f"\n{response}\n")

            except KeyboardInterrupt:
                print("\n\nSession ended.")
                break
            except Exception as e:
                print(f"Error: {e}")

    elif args.command == "summarize":
        summary = tracker.compute_daily_summary(
            model_name=args.model_name,
            model_type=args.model,
            date=args.date
        )

        if summary:
            print(f"\nDaily Summary for {summary.date}")
            print("-" * 40)
            print(f"  Total Requests: {summary.total_requests}")
            print(f"  Successful: {summary.successful_requests}")
            print(f"  Error Rate: {summary.error_rate:.2f}%")
            print(f"  Avg Latency: {summary.avg_latency_ms:.2f} ms")
            print(f"  p95 Latency: {summary.p95_latency_ms:.2f} ms")
            print(f"  Throughput: {summary.avg_throughput_tps:.2f} tokens/sec")
            print()
        else:
            print(f"No data found for {args.date or 'today'}")


if __name__ == "__main__":
    main()
