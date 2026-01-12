#!/usr/bin/env python3
"""
L4D2 SourcePawn Model Comparison Dashboard

A Streamlit-based dashboard for comparing model benchmark results,
viewing detailed test case analyses, and running live model comparisons.

Usage:
    streamlit run scripts/evaluation/model_dashboard.py

Features:
    - Load and display benchmark results from results/*.json
    - Overall pass rates by model
    - Category and difficulty breakdown charts
    - Individual test case results (expandable)
    - Side-by-side code comparison
    - Filter by category, difficulty, pass/fail status
    - Search test cases
    - Export to CSV and Markdown
    - Live testing mode for comparing models in real-time
"""

import json
import os
import subprocess
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_text, safe_read_json

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Check for Streamlit
try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

# Optional dependencies for charts
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =============================================================================
# DATA LOADING
# =============================================================================

def load_benchmark_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all benchmark result JSON files from directory."""
    results = {}
    if not results_dir.exists():
        return results

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Use model_name or filename as key
                key = data.get("model_name", json_file.stem)
                results[key] = {
                    "file": json_file.name,
                    "data": data
                }
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Failed to load {json_file.name}: {e}")

    return results


def get_test_cases_from_results(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Extract unique test cases from results."""
    test_cases = {}
    for model_name, result in results.items():
        for test_result in result["data"].get("test_results", []):
            test_id = test_result.get("test_id", "")
            if test_id and test_id not in test_cases:
                test_cases[test_id] = {
                    "id": test_id,
                    "results": {}
                }
            if test_id:
                test_cases[test_id]["results"][model_name] = test_result
    return test_cases


# =============================================================================
# CHART HELPERS
# =============================================================================

def create_pass_rate_chart(results: Dict[str, Dict]) -> Optional[Any]:
    """Create bar chart for overall pass rates."""
    if not PLOTLY_AVAILABLE or not results:
        return None

    models = []
    pass_rates = []
    avg_scores = []

    for model_name, result in results.items():
        data = result["data"]
        models.append(model_name[:30])  # Truncate long names
        pass_rates.append(data.get("pass_rate", 0))
        avg_scores.append(data.get("average_score", 0))

    fig = go.Figure(data=[
        go.Bar(name="Pass Rate (%)", x=models, y=pass_rates, marker_color="green"),
        go.Bar(name="Avg Score", x=models, y=avg_scores, marker_color="blue"),
    ])
    fig.update_layout(
        title="Overall Model Performance",
        xaxis_title="Model",
        yaxis_title="Value",
        barmode="group",
        height=400
    )
    return fig


def create_category_chart(results: Dict[str, Dict]) -> Optional[Any]:
    """Create grouped bar chart for category breakdown."""
    if not PLOTLY_AVAILABLE or not results:
        return None

    # Get all categories
    categories = set()
    for result in results.values():
        categories.update(result["data"].get("by_category", {}).keys())
    categories = sorted(categories)

    if not categories:
        return None

    fig = go.Figure()

    for model_name, result in results.items():
        by_category = result["data"].get("by_category", {})
        values = [by_category.get(cat, {}).get("pass_rate", 0) for cat in categories]
        fig.add_trace(go.Bar(name=model_name[:20], x=categories, y=values))

    fig.update_layout(
        title="Pass Rate by Category",
        xaxis_title="Category",
        yaxis_title="Pass Rate (%)",
        barmode="group",
        height=400
    )
    return fig


def create_difficulty_chart(results: Dict[str, Dict]) -> Optional[Any]:
    """Create grouped bar chart for difficulty breakdown."""
    if not PLOTLY_AVAILABLE or not results:
        return None

    difficulties = ["easy", "medium", "hard"]

    fig = go.Figure()

    for model_name, result in results.items():
        by_difficulty = result["data"].get("by_difficulty", {})
        values = [by_difficulty.get(diff, {}).get("pass_rate", 0) for diff in difficulties]
        fig.add_trace(go.Bar(name=model_name[:20], x=difficulties, y=values))

    fig.update_layout(
        title="Pass Rate by Difficulty",
        xaxis_title="Difficulty",
        yaxis_title="Pass Rate (%)",
        barmode="group",
        height=400
    )
    return fig


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_csv(results: Dict[str, Dict]) -> str:
    """Export comparison data to CSV format."""
    lines = ["Model,Total Tests,Passed,Failed,Pass Rate (%),Average Score,Execution Time (s)"]

    for model_name, result in results.items():
        data = result["data"]
        lines.append(
            f'"{model_name}",{data.get("total_tests", 0)},{data.get("passed", 0)},'
            f'{data.get("failed", 0)},{data.get("pass_rate", 0):.1f},'
            f'{data.get("average_score", 0):.2f},{data.get("execution_time", 0):.1f}'
        )

    return "\n".join(lines)


def export_to_markdown(results: Dict[str, Dict]) -> str:
    """Export comparison data to Markdown format."""
    lines = [
        f"# L4D2 Model Comparison Report",
        f"",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Overall Comparison",
        f"",
        f"| Model | Tests | Passed | Failed | Pass Rate | Avg Score | Time |",
        f"|-------|-------|--------|--------|-----------|-----------|------|",
    ]

    for model_name, result in results.items():
        data = result["data"]
        lines.append(
            f"| {model_name} | {data.get('total_tests', 0)} | {data.get('passed', 0)} | "
            f"{data.get('failed', 0)} | {data.get('pass_rate', 0):.1f}% | "
            f"{data.get('average_score', 0):.2f} | {data.get('execution_time', 0):.1f}s |"
        )

    # Add category breakdown
    lines.extend([
        "",
        "## Results by Category",
        ""
    ])

    # Get all categories
    categories = set()
    for result in results.values():
        categories.update(result["data"].get("by_category", {}).keys())

    for category in sorted(categories):
        lines.extend([
            f"### {category}",
            "",
            "| Model | Pass Rate | Avg Score |",
            "|-------|-----------|-----------|",
        ])
        for model_name, result in results.items():
            cat_data = result["data"].get("by_category", {}).get(category, {})
            lines.append(
                f"| {model_name} | {cat_data.get('pass_rate', 0):.1f}% | "
                f"{cat_data.get('avg_score', 0):.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# LIVE TESTING
# =============================================================================

class ModelClient:
    """Base client for model inference."""

    def generate(self, prompt: str, system: str = None) -> Tuple[str, float]:
        """Generate response. Returns (response, execution_time)."""
        raise NotImplementedError


class OllamaClient(ModelClient):
    """Client for Ollama local inference."""

    def __init__(self, model: str = "l4d2-code-v10plus"):
        self.model = model
        self._check_available()

    def _check_available(self) -> bool:
        """Check if Ollama and model are available."""
        if not shutil.which("ollama"):
            raise RuntimeError("Ollama not found")
        return True

    def is_model_available(self) -> bool:
        """Check if model is available in Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return self.model in result.stdout
        except Exception:
            return False

    def generate(self, prompt: str, system: str = None) -> Tuple[str, float]:
        """Generate completion using Ollama."""
        import time

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        start_time = time.time()
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, full_prompt],
                capture_output=True,
                text=True,
                timeout=180
            )
            execution_time = time.time() - start_time
            return result.stdout.strip(), execution_time
        except subprocess.TimeoutExpired:
            return "Error: Generation timed out", time.time() - start_time
        except Exception as e:
            return f"Error: {e}", time.time() - start_time


class OpenAIClient(ModelClient):
    """Client for OpenAI API."""

    def __init__(self, model_id: str, api_key: str = None):
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise RuntimeError("openai package not installed")

    def generate(self, prompt: str, system: str = None) -> Tuple[str, float]:
        """Generate completion using OpenAI."""
        import time

        if not system:
            system = "You are an expert SourcePawn developer for Left 4 Dead 2."

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.3
            )
            execution_time = time.time() - start_time
            return response.choices[0].message.content, execution_time
        except Exception as e:
            return f"Error: {e}", time.time() - start_time


def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            return [line.split()[0] for line in lines if line.strip()]
    except Exception:
        pass
    return []


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="L4D2 Model Comparison Dashboard",
        page_icon="🎮",
        layout="wide"
    )

    st.title("L4D2 SourcePawn Model Comparison Dashboard")

    # Sidebar for navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Test Results", "Live Testing", "Export"]
    )

    # Load results
    results_dir = PROJECT_ROOT / "results"
    results = load_benchmark_results(results_dir)

    if page == "Overview":
        render_overview(results)
    elif page == "Test Results":
        render_test_results(results)
    elif page == "Live Testing":
        render_live_testing()
    elif page == "Export":
        render_export(results)


def render_overview(results: Dict[str, Dict]):
    """Render the overview page with charts and summaries."""
    st.header("Model Performance Overview")

    if not results:
        st.warning("No benchmark results found in results/ directory.")
        st.info("Run a benchmark first: `python scripts/evaluation/benchmark_suite.py`")

        # Show sample data structure
        st.subheader("Expected JSON Format")
        sample = {
            "model_name": "example-model",
            "model_type": "ollama",
            "timestamp": "2024-01-01T12:00:00",
            "total_tests": 55,
            "passed": 40,
            "failed": 15,
            "pass_rate": 72.7,
            "average_score": 7.5,
            "by_category": {
                "basic_syntax": {"total": 10, "passed": 8, "pass_rate": 80.0},
            },
            "by_difficulty": {
                "easy": {"total": 15, "passed": 13, "pass_rate": 86.7},
            },
            "test_results": []
        }
        st.json(sample)
        return

    # Summary metrics
    st.subheader("Summary")
    cols = st.columns(len(results))

    for i, (model_name, result) in enumerate(results.items()):
        data = result["data"]
        with cols[i]:
            st.metric(
                label=model_name[:25],
                value=f"{data.get('pass_rate', 0):.1f}%",
                delta=f"Avg: {data.get('average_score', 0):.2f}"
            )

    # Overall pass rate chart
    st.subheader("Overall Performance")
    fig = create_pass_rate_chart(results)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        if not PLOTLY_AVAILABLE:
            st.info("Install plotly for charts: pip install plotly")
        else:
            st.info("No data to display")

    # Category breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("By Category")
        fig = create_category_chart(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback table
            for model_name, result in results.items():
                st.write(f"**{model_name}**")
                by_cat = result["data"].get("by_category", {})
                for cat, stats in by_cat.items():
                    st.write(f"  - {cat}: {stats.get('pass_rate', 0):.1f}%")

    with col2:
        st.subheader("By Difficulty")
        fig = create_difficulty_chart(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback table
            for model_name, result in results.items():
                st.write(f"**{model_name}**")
                by_diff = result["data"].get("by_difficulty", {})
                for diff, stats in by_diff.items():
                    st.write(f"  - {diff}: {stats.get('pass_rate', 0):.1f}%")

    # Detailed comparison table
    st.subheader("Detailed Comparison")

    if PANDAS_AVAILABLE:
        table_data = []
        for model_name, result in results.items():
            data = result["data"]
            table_data.append({
                "Model": model_name,
                "Total Tests": data.get("total_tests", 0),
                "Passed": data.get("passed", 0),
                "Failed": data.get("failed", 0),
                "Pass Rate (%)": f"{data.get('pass_rate', 0):.1f}",
                "Avg Score": f"{data.get('average_score', 0):.2f}",
                "Time (s)": f"{data.get('execution_time', 0):.1f}",
            })
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    else:
        # Fallback without pandas
        for model_name, result in results.items():
            data = result["data"]
            st.write(
                f"**{model_name}**: {data.get('passed', 0)}/{data.get('total_tests', 0)} "
                f"({data.get('pass_rate', 0):.1f}%) - Avg: {data.get('average_score', 0):.2f}"
            )


def render_test_results(results: Dict[str, Dict]):
    """Render the test results page with filtering and details."""
    st.header("Individual Test Results")

    if not results:
        st.warning("No benchmark results found.")
        return

    # Get all test cases
    test_cases = get_test_cases_from_results(results)

    if not test_cases:
        st.warning("No test results found in loaded data.")
        return

    # Filters
    st.subheader("Filters")
    col1, col2, col3, col4 = st.columns(4)

    # Get filter options from data
    categories = set()
    difficulties = set()

    for model_name, result in results.items():
        for test_result in result["data"].get("test_results", []):
            # Try to determine category from test_id
            test_id = test_result.get("test_id", "")
            if test_id:
                if test_id.startswith("syntax_"):
                    categories.add("basic_syntax")
                elif test_id.startswith("api_"):
                    categories.add("l4d2_api")
                elif test_id.startswith("event_"):
                    categories.add("event_handling")
                elif test_id.startswith("si_"):
                    categories.add("special_infected")
                elif test_id.startswith("adv_"):
                    categories.add("advanced_patterns")

    difficulties = {"easy", "medium", "hard"}

    with col1:
        selected_category = st.selectbox(
            "Category",
            ["All"] + sorted(categories)
        )

    with col2:
        selected_difficulty = st.selectbox(
            "Difficulty",
            ["All"] + sorted(difficulties)
        )

    with col3:
        selected_status = st.selectbox(
            "Status",
            ["All", "Passed", "Failed"]
        )

    with col4:
        search_query = st.text_input("Search Test ID", "")

    # Model selection for comparison
    model_names = list(results.keys())
    if len(model_names) > 1:
        selected_models = st.multiselect(
            "Compare Models",
            model_names,
            default=model_names[:2]
        )
    else:
        selected_models = model_names

    # Filter and display test cases
    st.subheader("Test Cases")

    for test_id, test_case in sorted(test_cases.items()):
        # Apply filters
        if search_query and search_query.lower() not in test_id.lower():
            continue

        # Category filter
        if selected_category != "All":
            if selected_category == "basic_syntax" and not test_id.startswith("syntax_"):
                continue
            if selected_category == "l4d2_api" and not test_id.startswith("api_"):
                continue
            if selected_category == "event_handling" and not test_id.startswith("event_"):
                continue
            if selected_category == "special_infected" and not test_id.startswith("si_"):
                continue
            if selected_category == "advanced_patterns" and not test_id.startswith("adv_"):
                continue

        # Status filter
        if selected_status != "All":
            has_matching_status = False
            for model_name in selected_models:
                if model_name in test_case["results"]:
                    passed = test_case["results"][model_name].get("passed", False)
                    if (selected_status == "Passed" and passed) or (selected_status == "Failed" and not passed):
                        has_matching_status = True
                        break
            if not has_matching_status:
                continue

        # Create expandable section for each test
        with st.expander(f"{test_id}", expanded=False):
            # Show results for each selected model
            cols = st.columns(len(selected_models))

            for i, model_name in enumerate(selected_models):
                with cols[i]:
                    st.write(f"**{model_name[:20]}**")

                    if model_name not in test_case["results"]:
                        st.write("No result")
                        continue

                    result = test_case["results"][model_name]
                    passed = result.get("passed", False)
                    score = result.get("score", 0)

                    # Status badge
                    if passed:
                        st.success(f"PASS - Score: {score:.1f}")
                    else:
                        st.error(f"FAIL - Score: {score:.1f}")

                    # Details
                    st.write(f"- Code lines: {result.get('code_lines', 0)}")
                    st.write(f"- Has includes: {result.get('has_includes', False)}")
                    st.write(f"- Time: {result.get('execution_time', 0):.2f}s")

                    # Expected/missing patterns
                    expected = result.get("expected_found", [])
                    missing = result.get("expected_missing", [])
                    forbidden = result.get("forbidden_found", [])

                    if expected:
                        st.write(f"Found: {', '.join(expected)}")
                    if missing:
                        st.write(f"Missing: {', '.join(missing)}")
                    if forbidden:
                        st.write(f"Forbidden: {', '.join(forbidden)}")

                    # Show response
                    response = result.get("response", "")
                    if response:
                        st.code(response[:2000], language="c")


def render_live_testing():
    """Render the live testing page."""
    st.header("Live Model Testing")

    st.info(
        "Test models in real-time by entering a prompt and comparing outputs side by side."
    )

    # Model configuration
    st.subheader("Configure Models")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model 1: Ollama**")
        ollama_models = get_available_ollama_models()
        if ollama_models:
            model1_name = st.selectbox(
                "Select Ollama model",
                ollama_models,
                key="ollama_model"
            )
            model1_enabled = st.checkbox("Enable Model 1", value=True)
        else:
            st.warning("No Ollama models found. Install Ollama and run `ollama pull <model>`")
            model1_enabled = False
            model1_name = None

    with col2:
        st.write("**Model 2: OpenAI**")
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_api_key:
            model2_id = st.text_input(
                "OpenAI Model ID",
                value="gpt-4o-mini",
                key="openai_model"
            )
            model2_enabled = st.checkbox("Enable Model 2", value=True)
        else:
            st.warning("OPENAI_API_KEY not set")
            model2_enabled = False
            model2_id = None

    # System prompt
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "System prompt for models",
        value=(
            "You are an expert SourcePawn and VScript developer for Left 4 Dead 2 SourceMod plugins. "
            "Write clean, well-documented code with proper error handling. "
            "CRITICAL: Use GetRandomFloat() NOT RandomFloat(). Use GetRandomInt() NOT RandomInt()."
        ),
        height=100
    )

    # Prompt input
    st.subheader("Test Prompt")
    test_prompt = st.text_area(
        "Enter your prompt",
        value="Write a SourcePawn plugin that heals all survivors when a player types !healall in chat.",
        height=100
    )

    # Run button
    if st.button("Run Comparison", type="primary"):
        if not test_prompt:
            st.warning("Please enter a prompt")
            return

        col1, col2 = st.columns(2)

        # Model 1 (Ollama)
        with col1:
            st.write("**Model 1: Ollama**")
            if model1_enabled and model1_name:
                with st.spinner(f"Generating with {model1_name}..."):
                    try:
                        client = OllamaClient(model1_name)
                        response, exec_time = client.generate(test_prompt, system_prompt)
                        st.write(f"Time: {exec_time:.2f}s")
                        st.code(response, language="c")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Model 1 disabled")

        # Model 2 (OpenAI)
        with col2:
            st.write("**Model 2: OpenAI**")
            if model2_enabled and model2_id:
                with st.spinner(f"Generating with {model2_id}..."):
                    try:
                        client = OpenAIClient(model2_id)
                        response, exec_time = client.generate(test_prompt, system_prompt)
                        st.write(f"Time: {exec_time:.2f}s")
                        st.code(response, language="c")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Model 2 disabled")

    # Sample prompts
    st.subheader("Sample Prompts")
    sample_prompts = [
        "Write a SourcePawn plugin that heals all survivors when a player types !healall in chat.",
        "Write a function that increases a survivor's movement speed by 30%.",
        "Write a plugin that announces when a Tank spawns and displays its health.",
        "Write a plugin that prevents friendly fire damage between survivors.",
        "Write a function that teleports a player to specified coordinates.",
        "Write a plugin that tracks Hunter pounce damage using the correct event.",
        "Write a plugin that detects when a survivor gets covered in Boomer bile.",
        "Write a plugin that creates a repeating timer every 5 seconds.",
    ]

    for prompt in sample_prompts:
        if st.button(prompt[:60] + "...", key=f"sample_{hash(prompt)}"):
            st.session_state["test_prompt"] = prompt
            st.experimental_rerun()


def render_export(results: Dict[str, Dict]):
    """Render the export page."""
    st.header("Export Results")

    if not results:
        st.warning("No benchmark results to export.")
        return

    st.subheader("Export Options")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**CSV Export**")
        csv_content = export_to_csv(results)
        st.download_button(
            label="Download CSV",
            data=csv_content,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        st.text_area("CSV Preview", csv_content, height=200)

    with col2:
        st.write("**Markdown Export**")
        md_content = export_to_markdown(results)
        st.download_button(
            label="Download Markdown",
            data=md_content,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
        st.text_area("Markdown Preview", md_content, height=200)

    # Save to file option
    st.subheader("Save to Project")

    save_filename = st.text_input(
        "Filename (in docs/)",
        value=f"model_comparison_{datetime.now().strftime('%Y%m%d')}.md"
    )

    if st.button("Save to docs/"):
        try:
            output_path = safe_path(f"docs/{save_filename}", PROJECT_ROOT, create_parents=True)
            safe_write_text(str(output_path), md_content, PROJECT_ROOT)
            st.success(f"Saved to {output_path}")
        except Exception as e:
            st.error(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
