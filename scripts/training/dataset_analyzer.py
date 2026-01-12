#!/usr/bin/env python3
"""
Dataset Quality Analyzer for L4D2 Training Data

Comprehensive analysis of training data quality including:
- Token count distribution
- Prompt/response length analysis
- Duplicate detection
- Language distribution (SourcePawn vs VScript)
- Category coverage
- Complexity scoring
- Visual reports (histograms, word clouds, heatmaps)
- Improvement recommendations

Usage:
    python dataset_analyzer.py --data l4d2_train_v15.jsonl --report
    python dataset_analyzer.py --data l4d2_train_v15.jsonl --output analysis_report.json
"""

import sys
import json
import re
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
import statistics

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_json, safe_read_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "analysis_reports"

# Language detection patterns
SOURCEPAWN_PATTERNS = [
    r'#include\s*<sourcemod>',
    r'#pragma\s+semicolon',
    r'#pragma\s+newdecls',
    r'public\s+Plugin\s+myinfo',
    r'public\s+void\s+OnPluginStart',
    r'RegConsoleCmd|RegAdminCmd',
    r'CreateConVar',
    r'HookEvent\s*\(',
    r'GetClientTeam|GetClientHealth',
    r'PrintToChat|PrintToChatAll',
    r'SourceMod|SourcePawn',
    r'Action\s+\w+\s*\(.*Handle\s+timer',
    r'stock\s+\w+\s+\w+\s*\(',
    r'methodmap\s+\w+',
    r'enum\s+struct',
]

VSCRIPT_PATTERNS = [
    r'DirectorOptions\s*<-',
    r'function\s+\w+\s*\(',
    r'Entities\.Find|Entities\.First',
    r'::VSLib\.',
    r'foreach\s*\(',
    r'ZSpawn|SpawnZombie',
    r'NetProps\.GetProp',
    r'DoEntFire',
    r'local\s+\w+\s*=',
    r'<-\s*\{',
    r'Msg\s*\(',
    r'Director\.GetCommonInfectedCount',
    r'::EMS',
    r'ScriptPrintMessage',
]

# Category detection patterns
CATEGORY_PATTERNS = {
    'spawn_infected': [
        r'SpawnZombie|ZSpawn|SpawnSpecial|SpawnWitch|SpawnTank',
        r'SpawnBoomer|SpawnSmoker|SpawnHunter|SpawnJockey|SpawnCharger|SpawnSpitter',
        r'Infected_.*Spawn|infected.*spawn|SpawnInfected',
    ],
    'player_health': [
        r'GetClient(Health|MaxHealth)|SetEntity(Health|MaxHealth)',
        r'HealSurvivor|ReviveSurvivor|IncapSurvivor',
        r'IsPlayerIncapped|IsPlayerAlive|IsClientIncapacitated',
        r'Health|health.*heal|heal.*player',
    ],
    'weapons': [
        r'GivePlayerItem|RemovePlayerItem|EquipPlayerWeapon',
        r'GetPlayerWeapon|GetEntProp.*Weapon',
        r'weapon_.*give|give.*weapon',
        r'melee|rifle|shotgun|smg|pistol|magnum',
    ],
    'events': [
        r'HookEvent\s*\(|EventHook|OnEvent',
        r'player_death|player_spawn|player_hurt',
        r'round_start|round_end|map_transition',
        r'witch_spawn|tank_spawn|infected_death',
    ],
    'commands': [
        r'RegConsoleCmd|RegAdminCmd|AddCommandListener',
        r'sm_\w+|Command_\w+',
        r'ReplyToCommand|GetCmdArg',
    ],
    'timers': [
        r'CreateTimer|CreateDataTimer|KillTimer',
        r'Timer_\w+|Handle\s+timer',
        r'TIMER_REPEAT|TIMER_FLAG_NO_MAPCHANGE',
    ],
    'entities': [
        r'CreateEntityByName|DispatchSpawn|AcceptEntityInput',
        r'GetEntProp|SetEntProp|GetEntData|SetEntData',
        r'TeleportEntity|RemoveEntity',
        r'Entities\.|entity_',
    ],
    'director': [
        r'DirectorOptions|DirectorScript',
        r'L4D_.*Director|Director\.',
        r'CommonLimit|MobMax|MobMin|WanderingZombieDensity',
        r'SpecialRespawn|TankLimit',
    ],
    'convars': [
        r'CreateConVar|FindConVar|SetConVarInt|GetConVarInt',
        r'ConVar\s+\w+|cvar|g_cv\w+',
        r'sm_.*_enabled|sm_.*_limit',
    ],
    'menus': [
        r'Menu\s+\w+\s*=\s*new\s+Menu|CreateMenu|MenuHandler',
        r'menu\.AddItem|menu\.Display|MenuAction',
        r'TopMenu|Panel',
    ],
    'database': [
        r'SQL_Connect|SQL_Query|SQL_FetchRow',
        r'Database\s+\w+|DBDriver',
        r'mysql|sqlite|database',
    ],
    'versus': [
        r'versus|survival|scavenge|coop',
        r'survivor.*vs.*infected|infected.*team',
        r'GetSurvivor|GetInfected',
    ],
    'admin': [
        r'AdminFlag|ADMFLAG|CheckCommandAccess|GetAdminFlag',
        r'IsClientAdmin|GetUserAdmin|admin_',
        r'ban|kick|mute|gag',
    ],
    'special_infected': [
        r'tank|boomer|smoker|hunter|spitter|jockey|charger|witch',
        r'SI_|Special\s*Infected|GetZombieClass',
        r'Infected.*Class|CLASS_',
    ],
}

# Complexity scoring weights
COMPLEXITY_WEIGHTS = {
    'lines': 0.1,
    'functions': 2.0,
    'conditionals': 0.5,
    'loops': 0.7,
    'includes': 0.3,
    'comments': 0.2,
    'events': 1.0,
    'timers': 0.8,
    'nested_depth': 1.5,
}


@dataclass
class SampleAnalysis:
    """Analysis of a single training sample."""
    index: int
    system_length: int
    prompt_length: int
    response_length: int
    total_length: int
    prompt_tokens_est: int
    response_tokens_est: int
    total_tokens_est: int
    language: str
    categories: List[str]
    complexity_score: float
    content_hash: str
    prompt_hash: str
    quality_issues: List[str]


@dataclass
class DuplicateGroup:
    """Group of duplicate or near-duplicate samples."""
    hash_value: str
    indices: List[int]
    sample_preview: str
    duplicate_type: str  # 'exact' or 'near'


@dataclass
class DatasetReport:
    """Complete analysis report for a dataset."""
    file_name: str
    analysis_date: str
    total_samples: int

    # Length statistics
    prompt_length_stats: Dict[str, float] = field(default_factory=dict)
    response_length_stats: Dict[str, float] = field(default_factory=dict)
    total_length_stats: Dict[str, float] = field(default_factory=dict)

    # Token statistics
    prompt_token_stats: Dict[str, float] = field(default_factory=dict)
    response_token_stats: Dict[str, float] = field(default_factory=dict)
    total_token_stats: Dict[str, float] = field(default_factory=dict)

    # Distribution
    token_distribution: Dict[str, int] = field(default_factory=dict)
    length_distribution: Dict[str, int] = field(default_factory=dict)

    # Language distribution
    language_distribution: Dict[str, int] = field(default_factory=dict)

    # Category coverage
    category_coverage: Dict[str, int] = field(default_factory=dict)
    category_combinations: Dict[str, int] = field(default_factory=dict)

    # Complexity
    complexity_stats: Dict[str, float] = field(default_factory=dict)
    complexity_distribution: Dict[str, int] = field(default_factory=dict)

    # Duplicates
    exact_duplicates: int = 0
    near_duplicates: int = 0
    duplicate_groups: List[Dict] = field(default_factory=list)

    # Quality issues
    quality_issues: Dict[str, int] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Sample analyses (optional - can be very large)
    samples: List[Dict] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count without external dependencies.
    Uses a simple heuristic: ~4 characters per token for code.
    """
    # Remove extra whitespace
    cleaned = ' '.join(text.split())

    # Code tends to have more tokens per character due to operators, keywords
    # Rough estimate: 1 token per 3.5 characters for code
    return max(1, len(cleaned) // 4 + len(cleaned.split()) // 2)


def detect_language(text: str) -> str:
    """Detect whether code is SourcePawn or VScript."""
    sp_score = 0
    vs_score = 0

    for pattern in SOURCEPAWN_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            sp_score += 1

    for pattern in VSCRIPT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            vs_score += 1

    if sp_score > vs_score:
        return 'sourcepawn'
    elif vs_score > sp_score:
        return 'vscript'
    elif sp_score > 0:
        return 'sourcepawn'  # Default to SP if any matches
    else:
        return 'unknown'


def detect_categories(text: str) -> List[str]:
    """Detect categories present in the code."""
    categories = []

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                categories.append(category)
                break

    return categories if categories else ['uncategorized']


def calculate_complexity(text: str) -> float:
    """Calculate code complexity score."""
    scores = {}

    # Line count
    lines = text.count('\n') + 1
    scores['lines'] = min(lines / 50, 2.0)  # Cap at 2.0

    # Function count
    func_matches = len(re.findall(r'(?:public|stock|static|function)\s+\w+\s+\w+\s*\(', text))
    scores['functions'] = min(func_matches / 3, 2.0)

    # Conditional count
    cond_matches = len(re.findall(r'\b(if|else|switch|case)\b', text))
    scores['conditionals'] = min(cond_matches / 5, 2.0)

    # Loop count
    loop_matches = len(re.findall(r'\b(for|while|foreach|do)\b', text))
    scores['loops'] = min(loop_matches / 3, 2.0)

    # Include count
    include_matches = len(re.findall(r'#include\s*[<"]', text))
    scores['includes'] = min(include_matches / 5, 2.0)

    # Comment presence
    comment_matches = len(re.findall(r'//|/\*|\*/', text))
    scores['comments'] = min(comment_matches / 10, 1.0)

    # Event hooks
    event_matches = len(re.findall(r'HookEvent|OnEvent|\bevent_', text, re.IGNORECASE))
    scores['events'] = min(event_matches / 2, 2.0)

    # Timer usage
    timer_matches = len(re.findall(r'CreateTimer|Timer_', text))
    scores['timers'] = min(timer_matches / 2, 2.0)

    # Nested depth (rough estimate based on brace depth)
    max_depth = 0
    current_depth = 0
    for char in text:
        if char == '{':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == '}':
            current_depth = max(0, current_depth - 1)
    scores['nested_depth'] = min(max_depth / 4, 2.0)

    # Calculate weighted score
    total = 0
    for key, value in scores.items():
        total += value * COMPLEXITY_WEIGHTS.get(key, 1.0)

    # Normalize to 0-10 scale
    return round(min(total / 2, 10.0), 2)


def detect_quality_issues(prompt: str, response: str) -> List[str]:
    """Detect potential quality issues in a sample."""
    issues = []

    # Check prompt issues
    if len(prompt) < 20:
        issues.append('prompt_too_short')
    if len(prompt) > 500:
        issues.append('prompt_too_long')
    if prompt.strip().startswith('//'):
        issues.append('prompt_is_comment')
    if not any(c.isalpha() for c in prompt):
        issues.append('prompt_no_alpha')

    # Check response issues
    if len(response) < 50:
        issues.append('response_too_short')
    if len(response) > 10000:
        issues.append('response_too_long')

    # Check for incomplete code
    if re.search(r'TODO|FIXME|XXX|HACK', response, re.IGNORECASE):
        issues.append('contains_todo')
    if response.count('{') != response.count('}'):
        issues.append('unbalanced_braces')
    if response.count('(') != response.count(')'):
        issues.append('unbalanced_parens')

    # Check for sensitive content
    if re.search(r'password|api[_-]?key|secret|token\s*=', response, re.IGNORECASE):
        issues.append('possible_sensitive')

    # Check for syntax issues
    if re.search(r';\s*;', response):
        issues.append('double_semicolon')

    # Check for low quality patterns
    if response.count('\n\n\n') > 2:
        issues.append('excessive_blank_lines')
    if len(set(response.split('\n'))) < len(response.split('\n')) * 0.5:
        issues.append('repetitive_lines')

    return issues


def compute_hash(text: str, truncate: int = 0) -> str:
    """Compute hash of text for duplicate detection."""
    # Normalize whitespace
    normalized = ' '.join(text.lower().split())
    if truncate > 0:
        normalized = normalized[:truncate]
    return hashlib.md5(normalized.encode()).hexdigest()


def analyze_sample(index: int, sample: Dict) -> SampleAnalysis:
    """Analyze a single training sample."""
    messages = sample.get('messages', [])

    # Extract content
    system_content = ''
    prompt_content = ''
    response_content = ''

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if role == 'system':
            system_content = content
        elif role == 'user':
            prompt_content = content
        elif role == 'assistant':
            response_content = content

    # Calculate lengths
    system_length = len(system_content)
    prompt_length = len(prompt_content)
    response_length = len(response_content)
    total_length = system_length + prompt_length + response_length

    # Estimate tokens
    prompt_tokens = estimate_tokens(prompt_content)
    response_tokens = estimate_tokens(response_content)
    total_tokens = estimate_tokens(system_content) + prompt_tokens + response_tokens

    # Detect language and categories from response
    language = detect_language(response_content)
    categories = detect_categories(response_content)

    # Calculate complexity
    complexity = calculate_complexity(response_content)

    # Compute hashes for duplicate detection
    content_hash = compute_hash(response_content)
    prompt_hash = compute_hash(prompt_content)

    # Detect quality issues
    quality_issues = detect_quality_issues(prompt_content, response_content)

    return SampleAnalysis(
        index=index,
        system_length=system_length,
        prompt_length=prompt_length,
        response_length=response_length,
        total_length=total_length,
        prompt_tokens_est=prompt_tokens,
        response_tokens_est=response_tokens,
        total_tokens_est=total_tokens,
        language=language,
        categories=categories,
        complexity_score=complexity,
        content_hash=content_hash,
        prompt_hash=prompt_hash,
        quality_issues=quality_issues,
    )


def find_duplicates(analyses: List[SampleAnalysis]) -> Tuple[List[DuplicateGroup], List[DuplicateGroup]]:
    """Find exact and near duplicates in the dataset."""
    # Group by content hash for exact duplicates
    content_groups = defaultdict(list)
    for analysis in analyses:
        content_groups[analysis.content_hash].append(analysis.index)

    exact_duplicates = []
    for hash_val, indices in content_groups.items():
        if len(indices) > 1:
            exact_duplicates.append(DuplicateGroup(
                hash_value=hash_val,
                indices=indices,
                sample_preview=f"Sample {indices[0]}",
                duplicate_type='exact'
            ))

    # For near duplicates, use prompt hash
    prompt_groups = defaultdict(list)
    for analysis in analyses:
        prompt_groups[analysis.prompt_hash].append(analysis.index)

    near_duplicates = []
    for hash_val, indices in prompt_groups.items():
        if len(indices) > 1:
            # Check if they're not already exact duplicates
            if not all(analyses[i].content_hash == analyses[indices[0]].content_hash for i in indices):
                near_duplicates.append(DuplicateGroup(
                    hash_value=hash_val,
                    indices=indices,
                    sample_preview=f"Sample {indices[0]}",
                    duplicate_type='near'
                ))

    return exact_duplicates, near_duplicates


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate statistical summary of values."""
    if not values:
        return {
            'count': 0,
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'std': 0,
            'p25': 0,
            'p75': 0,
            'p90': 0,
            'p95': 0,
        }

    sorted_values = sorted(values)
    n = len(sorted_values)

    return {
        'count': n,
        'min': round(min(values), 2),
        'max': round(max(values), 2),
        'mean': round(statistics.mean(values), 2),
        'median': round(statistics.median(values), 2),
        'std': round(statistics.stdev(values), 2) if n > 1 else 0,
        'p25': round(sorted_values[int(n * 0.25)], 2),
        'p75': round(sorted_values[int(n * 0.75)], 2),
        'p90': round(sorted_values[int(n * 0.90)], 2),
        'p95': round(sorted_values[int(n * 0.95)], 2),
    }


def bucket_distribution(values: List[float], buckets: List[Tuple[float, float, str]]) -> Dict[str, int]:
    """Bucket values into distribution ranges."""
    distribution = {label: 0 for _, _, label in buckets}

    for value in values:
        for low, high, label in buckets:
            if low <= value < high:
                distribution[label] += 1
                break

    return distribution


def generate_recommendations(report: DatasetReport) -> List[str]:
    """Generate recommendations based on analysis."""
    recommendations = []

    # Check sample count
    if report.total_samples < 500:
        recommendations.append(
            f"LOW_SAMPLE_COUNT: Only {report.total_samples} samples. "
            "Consider adding more training data (target: 1000+ samples)."
        )

    # Check for imbalanced languages
    total = sum(report.language_distribution.values())
    for lang, count in report.language_distribution.items():
        ratio = count / total if total > 0 else 0
        if lang == 'vscript' and ratio < 0.15:
            recommendations.append(
                f"LOW_VSCRIPT_COVERAGE: Only {ratio:.1%} VScript samples. "
                "Consider adding more VScript/Squirrel training data."
            )
        if lang == 'unknown' and ratio > 0.1:
            recommendations.append(
                f"HIGH_UNKNOWN_LANGUAGE: {ratio:.1%} samples have unknown language. "
                "Review these samples for quality."
            )

    # Check category coverage
    uncategorized = report.category_coverage.get('uncategorized', 0)
    if uncategorized > report.total_samples * 0.2:
        recommendations.append(
            f"HIGH_UNCATEGORIZED: {uncategorized} samples ({uncategorized/report.total_samples:.1%}) "
            "are uncategorized. Consider improving category detection or adding more diverse samples."
        )

    # Check for underrepresented categories
    important_categories = ['spawn_infected', 'player_health', 'weapons', 'events', 'director']
    for cat in important_categories:
        count = report.category_coverage.get(cat, 0)
        if count < 20:
            recommendations.append(
                f"LOW_CATEGORY_{cat.upper()}: Only {count} samples for '{cat}'. "
                "Consider adding more examples for this category."
            )

    # Check duplicates
    if report.exact_duplicates > 10:
        recommendations.append(
            f"HIGH_DUPLICATES: {report.exact_duplicates} exact duplicate groups found. "
            "Consider deduplicating the dataset."
        )

    # Check length distribution
    if report.response_length_stats.get('median', 0) < 200:
        recommendations.append(
            "SHORT_RESPONSES: Median response length is low. "
            "Consider including more complete code examples."
        )
    if report.response_length_stats.get('p95', 0) > 5000:
        recommendations.append(
            "LONG_RESPONSES: Some responses are very long. "
            "Consider truncating or splitting very large code samples."
        )

    # Check quality issues
    total_issues = sum(report.quality_issues.values())
    if total_issues > report.total_samples * 0.3:
        recommendations.append(
            f"HIGH_QUALITY_ISSUES: {total_issues} quality issues detected. "
            "Review samples for incomplete code, TODO comments, or syntax errors."
        )

    # Check complexity distribution
    low_complexity = report.complexity_distribution.get('very_low', 0) + \
                    report.complexity_distribution.get('low', 0)
    if low_complexity > report.total_samples * 0.5:
        recommendations.append(
            f"LOW_COMPLEXITY: {low_complexity} samples have low complexity. "
            "Consider adding more sophisticated code examples."
        )

    # Check token distribution
    token_stats = report.total_token_stats
    if token_stats.get('p95', 0) > 2000:
        recommendations.append(
            "HIGH_TOKEN_COUNT: Some samples exceed 2000 tokens. "
            "Ensure your model context window can handle these."
        )

    if not recommendations:
        recommendations.append(
            "DATASET_QUALITY_GOOD: No major issues detected. "
            "The dataset appears well-balanced and suitable for training."
        )

    return recommendations


def generate_visual_reports(analyses: List[SampleAnalysis], report: DatasetReport,
                           output_dir: Path) -> Dict[str, str]:
    """Generate visual reports if matplotlib is available."""
    generated_files = {}

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available. Skipping visual reports.")
        return generated_files

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Token distribution histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    prompt_tokens = [a.prompt_tokens_est for a in analyses]
    response_tokens = [a.response_tokens_est for a in analyses]
    total_tokens = [a.total_tokens_est for a in analyses]

    axes[0].hist(prompt_tokens, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_title('Prompt Token Distribution')
    axes[0].set_xlabel('Tokens')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.median(prompt_tokens), color='red', linestyle='--', label=f'Median: {np.median(prompt_tokens):.0f}')
    axes[0].legend()

    axes[1].hist(response_tokens, bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
    axes[1].set_title('Response Token Distribution')
    axes[1].set_xlabel('Tokens')
    axes[1].axvline(np.median(response_tokens), color='red', linestyle='--', label=f'Median: {np.median(response_tokens):.0f}')
    axes[1].legend()

    axes[2].hist(total_tokens, bins=50, edgecolor='black', alpha=0.7, color='darkorange')
    axes[2].set_title('Total Token Distribution')
    axes[2].set_xlabel('Tokens')
    axes[2].axvline(np.median(total_tokens), color='red', linestyle='--', label=f'Median: {np.median(total_tokens):.0f}')
    axes[2].legend()

    plt.tight_layout()
    token_hist_path = output_dir / 'token_distribution.png'
    plt.savefig(token_hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    generated_files['token_distribution'] = str(token_hist_path)

    # 2. Language distribution pie chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    lang_labels = list(report.language_distribution.keys())
    lang_values = list(report.language_distribution.values())
    colors = ['#4CAF50', '#2196F3', '#9E9E9E']

    wedges, texts, autotexts = axes[0].pie(
        lang_values,
        labels=lang_labels,
        autopct='%1.1f%%',
        colors=colors[:len(lang_labels)],
        startangle=90
    )
    axes[0].set_title('Language Distribution')

    # 3. Category coverage bar chart
    cat_labels = list(report.category_coverage.keys())
    cat_values = list(report.category_coverage.values())

    # Sort by value
    sorted_pairs = sorted(zip(cat_values, cat_labels), reverse=True)
    cat_values, cat_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

    bars = axes[1].barh(cat_labels, cat_values, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Sample Count')
    axes[1].set_title('Category Coverage')
    axes[1].invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, cat_values):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', fontsize=8)

    plt.tight_layout()
    lang_cat_path = output_dir / 'language_category.png'
    plt.savefig(lang_cat_path, dpi=150, bbox_inches='tight')
    plt.close()
    generated_files['language_category'] = str(lang_cat_path)

    # 4. Complexity distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    complexity_scores = [a.complexity_score for a in analyses]

    axes[0].hist(complexity_scores, bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[0].set_title('Complexity Score Distribution')
    axes[0].set_xlabel('Complexity Score')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.median(complexity_scores), color='red', linestyle='--',
                   label=f'Median: {np.median(complexity_scores):.1f}')
    axes[0].legend()

    # Complexity vs response length scatter
    response_lengths = [a.response_length for a in analyses]
    axes[1].scatter(response_lengths, complexity_scores, alpha=0.3, s=10)
    axes[1].set_xlabel('Response Length (chars)')
    axes[1].set_ylabel('Complexity Score')
    axes[1].set_title('Complexity vs Response Length')

    plt.tight_layout()
    complexity_path = output_dir / 'complexity_analysis.png'
    plt.savefig(complexity_path, dpi=150, bbox_inches='tight')
    plt.close()
    generated_files['complexity_analysis'] = str(complexity_path)

    # 5. Quality issues heatmap
    if report.quality_issues:
        fig, ax = plt.subplots(figsize=(10, 6))

        issue_labels = list(report.quality_issues.keys())
        issue_values = list(report.quality_issues.values())

        # Sort by value
        sorted_pairs = sorted(zip(issue_values, issue_labels), reverse=True)
        issue_values, issue_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

        colors = plt.cm.Reds(np.array(issue_values) / max(issue_values) if issue_values else [])
        bars = ax.barh(issue_labels, issue_values, color=colors, edgecolor='black')
        ax.set_xlabel('Count')
        ax.set_title('Quality Issues Distribution')
        ax.invert_yaxis()

        for bar, val in zip(bars, issue_values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   str(val), va='center', fontsize=9)

        plt.tight_layout()
        quality_path = output_dir / 'quality_issues.png'
        plt.savefig(quality_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated_files['quality_issues'] = str(quality_path)

    # 6. Category co-occurrence heatmap
    try:
        categories = list(report.category_coverage.keys())
        if len(categories) > 1:
            n = len(categories)
            cooccurrence = np.zeros((n, n))

            for analysis in analyses:
                for i, cat1 in enumerate(categories):
                    for j, cat2 in enumerate(categories):
                        if cat1 in analysis.categories and cat2 in analysis.categories:
                            cooccurrence[i, j] += 1

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(cooccurrence, cmap='YlOrRd')

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(categories, fontsize=8)
            ax.set_title('Category Co-occurrence Heatmap')

            plt.colorbar(im, ax=ax, label='Co-occurrence Count')
            plt.tight_layout()
            heatmap_path = output_dir / 'category_heatmap.png'
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            generated_files['category_heatmap'] = str(heatmap_path)
    except Exception as e:
        logger.warning(f"Could not generate heatmap: {e}")

    # 7. Word cloud (if wordcloud is available)
    try:
        from wordcloud import WordCloud

        # Collect all response text
        all_text = ' '.join([a.content_hash for a in analyses])  # Use hashes as placeholder

        # Actually collect keywords from responses
        keywords = []
        for analysis in analyses:
            keywords.extend(analysis.categories)

        keyword_freq = Counter(keywords)

        if keyword_freq:
            wc = WordCloud(width=800, height=400, background_color='white',
                          colormap='viridis', max_words=100)
            wc.generate_from_frequencies(keyword_freq)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Category Word Cloud')

            wordcloud_path = output_dir / 'category_wordcloud.png'
            plt.savefig(wordcloud_path, dpi=150, bbox_inches='tight')
            plt.close()
            generated_files['category_wordcloud'] = str(wordcloud_path)
    except ImportError:
        logger.info("wordcloud not available. Skipping word cloud generation.")
    except Exception as e:
        logger.warning(f"Could not generate word cloud: {e}")

    return generated_files


def load_dataset(file_path: Path) -> List[Dict]:
    """Load JSONL dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_num}: {e}")
    return data


def analyze_dataset(file_path: Path, include_samples: bool = False) -> DatasetReport:
    """Perform complete dataset analysis."""
    logger.info(f"Loading dataset: {file_path}")
    data = load_dataset(file_path)

    logger.info(f"Analyzing {len(data)} samples...")

    # Analyze each sample
    analyses = []
    for i, sample in enumerate(data):
        analysis = analyze_sample(i, sample)
        analyses.append(analysis)

        if (i + 1) % 500 == 0:
            logger.info(f"  Analyzed {i + 1}/{len(data)} samples")

    # Calculate statistics
    prompt_lengths = [a.prompt_length for a in analyses]
    response_lengths = [a.response_length for a in analyses]
    total_lengths = [a.total_length for a in analyses]

    prompt_tokens = [a.prompt_tokens_est for a in analyses]
    response_tokens = [a.response_tokens_est for a in analyses]
    total_tokens = [a.total_tokens_est for a in analyses]

    complexity_scores = [a.complexity_score for a in analyses]

    # Token distribution buckets
    token_buckets = [
        (0, 100, '0-100'),
        (100, 250, '100-250'),
        (250, 500, '250-500'),
        (500, 1000, '500-1000'),
        (1000, 2000, '1000-2000'),
        (2000, 4000, '2000-4000'),
        (4000, float('inf'), '4000+'),
    ]

    # Length distribution buckets
    length_buckets = [
        (0, 500, '0-500'),
        (500, 1000, '500-1000'),
        (1000, 2000, '1000-2000'),
        (2000, 5000, '2000-5000'),
        (5000, 10000, '5000-10000'),
        (10000, float('inf'), '10000+'),
    ]

    # Complexity distribution buckets
    complexity_buckets = [
        (0, 2, 'very_low'),
        (2, 4, 'low'),
        (4, 6, 'medium'),
        (6, 8, 'high'),
        (8, float('inf'), 'very_high'),
    ]

    # Calculate language distribution
    language_dist = Counter(a.language for a in analyses)

    # Calculate category coverage
    category_counts = Counter()
    category_combos = Counter()
    for a in analyses:
        for cat in a.categories:
            category_counts[cat] += 1
        combo_key = '+'.join(sorted(a.categories))
        category_combos[combo_key] += 1

    # Find duplicates
    exact_dups, near_dups = find_duplicates(analyses)

    # Aggregate quality issues
    quality_issues = Counter()
    for a in analyses:
        for issue in a.quality_issues:
            quality_issues[issue] += 1

    # Build report
    report = DatasetReport(
        file_name=file_path.name,
        analysis_date=datetime.now().isoformat(),
        total_samples=len(data),

        prompt_length_stats=calculate_stats(prompt_lengths),
        response_length_stats=calculate_stats(response_lengths),
        total_length_stats=calculate_stats(total_lengths),

        prompt_token_stats=calculate_stats(prompt_tokens),
        response_token_stats=calculate_stats(response_tokens),
        total_token_stats=calculate_stats(total_tokens),

        token_distribution=bucket_distribution(total_tokens, token_buckets),
        length_distribution=bucket_distribution(total_lengths, length_buckets),

        language_distribution=dict(language_dist),

        category_coverage=dict(category_counts),
        category_combinations=dict(category_combos.most_common(20)),

        complexity_stats=calculate_stats(complexity_scores),
        complexity_distribution=bucket_distribution(complexity_scores, complexity_buckets),

        exact_duplicates=len(exact_dups),
        near_duplicates=len(near_dups),
        duplicate_groups=[asdict(d) for d in exact_dups[:10]],  # Top 10 only

        quality_issues=dict(quality_issues),
    )

    # Generate recommendations
    report.recommendations = generate_recommendations(report)

    # Include sample analyses if requested
    if include_samples:
        report.samples = [asdict(a) for a in analyses]

    return report, analyses


def print_report(report: DatasetReport) -> None:
    """Print formatted report to console."""
    print("\n" + "=" * 80)
    print("DATASET QUALITY ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nFile: {report.file_name}")
    print(f"Analysis Date: {report.analysis_date}")
    print(f"Total Samples: {report.total_samples:,}")

    print("\n" + "-" * 40)
    print("TOKEN STATISTICS")
    print("-" * 40)
    print(f"Prompt Tokens:   Mean={report.prompt_token_stats['mean']:.0f}, "
          f"Median={report.prompt_token_stats['median']:.0f}, "
          f"P95={report.prompt_token_stats['p95']:.0f}")
    print(f"Response Tokens: Mean={report.response_token_stats['mean']:.0f}, "
          f"Median={report.response_token_stats['median']:.0f}, "
          f"P95={report.response_token_stats['p95']:.0f}")
    print(f"Total Tokens:    Mean={report.total_token_stats['mean']:.0f}, "
          f"Median={report.total_token_stats['median']:.0f}, "
          f"P95={report.total_token_stats['p95']:.0f}")

    print("\nToken Distribution:")
    for bucket, count in report.token_distribution.items():
        pct = count / report.total_samples * 100
        bar = '#' * int(pct / 2)
        print(f"  {bucket:>10}: {count:>5} ({pct:>5.1f}%) {bar}")

    print("\n" + "-" * 40)
    print("LENGTH STATISTICS (characters)")
    print("-" * 40)
    print(f"Prompt Length:   Mean={report.prompt_length_stats['mean']:.0f}, "
          f"Median={report.prompt_length_stats['median']:.0f}, "
          f"Min={report.prompt_length_stats['min']:.0f}, "
          f"Max={report.prompt_length_stats['max']:.0f}")
    print(f"Response Length: Mean={report.response_length_stats['mean']:.0f}, "
          f"Median={report.response_length_stats['median']:.0f}, "
          f"Min={report.response_length_stats['min']:.0f}, "
          f"Max={report.response_length_stats['max']:.0f}")

    print("\n" + "-" * 40)
    print("LANGUAGE DISTRIBUTION")
    print("-" * 40)
    for lang, count in sorted(report.language_distribution.items(), key=lambda x: -x[1]):
        pct = count / report.total_samples * 100
        bar = '#' * int(pct / 2)
        print(f"  {lang:>12}: {count:>5} ({pct:>5.1f}%) {bar}")

    print("\n" + "-" * 40)
    print("CATEGORY COVERAGE")
    print("-" * 40)
    for cat, count in sorted(report.category_coverage.items(), key=lambda x: -x[1]):
        pct = count / report.total_samples * 100
        bar = '#' * int(pct / 2)
        print(f"  {cat:>20}: {count:>5} ({pct:>5.1f}%) {bar}")

    print("\n" + "-" * 40)
    print("COMPLEXITY ANALYSIS")
    print("-" * 40)
    print(f"Mean: {report.complexity_stats['mean']:.2f}, "
          f"Median: {report.complexity_stats['median']:.2f}, "
          f"Std: {report.complexity_stats['std']:.2f}")
    print("\nComplexity Distribution:")
    for level, count in report.complexity_distribution.items():
        pct = count / report.total_samples * 100
        bar = '#' * int(pct / 2)
        print(f"  {level:>10}: {count:>5} ({pct:>5.1f}%) {bar}")

    print("\n" + "-" * 40)
    print("DUPLICATE ANALYSIS")
    print("-" * 40)
    print(f"Exact duplicate groups: {report.exact_duplicates}")
    print(f"Near duplicate groups: {report.near_duplicates}")
    if report.duplicate_groups:
        print("\nTop duplicate groups:")
        for dg in report.duplicate_groups[:5]:
            print(f"  - {len(dg['indices'])} duplicates: samples {dg['indices'][:5]}{'...' if len(dg['indices']) > 5 else ''}")

    print("\n" + "-" * 40)
    print("QUALITY ISSUES")
    print("-" * 40)
    total_issues = sum(report.quality_issues.values())
    print(f"Total issues detected: {total_issues}")
    if report.quality_issues:
        for issue, count in sorted(report.quality_issues.items(), key=lambda x: -x[1]):
            pct = count / report.total_samples * 100
            print(f"  {issue:>25}: {count:>5} ({pct:>5.1f}%)")

    print("\n" + "-" * 40)
    print("RECOMMENDATIONS")
    print("-" * 40)
    for rec in report.recommendations:
        print(f"  * {rec}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training dataset quality for L4D2 AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_analyzer.py --data l4d2_train_v15.jsonl --report
  python dataset_analyzer.py --data data/processed/combined_train.jsonl --output report.json
  python dataset_analyzer.py --data l4d2_train_v15.jsonl --visual --output-dir analysis/
        """
    )

    parser.add_argument('--data', '-d', type=str, required=True,
                       help='Path to JSONL dataset file')
    parser.add_argument('--report', '-r', action='store_true',
                       help='Print detailed report to console')
    parser.add_argument('--output', '-o', type=str,
                       help='Save JSON report to file')
    parser.add_argument('--output-dir', type=str,
                       help='Directory for visual reports')
    parser.add_argument('--visual', '-v', action='store_true',
                       help='Generate visual reports (requires matplotlib)')
    parser.add_argument('--include-samples', action='store_true',
                       help='Include per-sample analysis in output (large)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Resolve data path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        # Try relative to processed dir first
        if (DATA_DIR / args.data).exists():
            data_path = DATA_DIR / args.data
        else:
            data_path = Path.cwd() / args.data

    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        sys.exit(1)

    # Validate path
    try:
        safe_path(str(data_path), PROJECT_ROOT)
    except ValueError:
        logger.error(f"Path traversal detected: {data_path}")
        sys.exit(1)

    # Run analysis
    report, analyses = analyze_dataset(data_path, include_samples=args.include_samples)

    # Print report
    if args.report or (not args.output and not args.visual):
        print_report(report)

    # Generate visual reports
    if args.visual:
        output_dir = Path(args.output_dir) if args.output_dir else REPORTS_DIR
        try:
            safe_path(str(output_dir), PROJECT_ROOT, create_parents=True)
        except ValueError:
            logger.error(f"Invalid output directory: {output_dir}")
            sys.exit(1)

        logger.info(f"Generating visual reports in {output_dir}")
        generated = generate_visual_reports(analyses, report, output_dir)

        if generated:
            print(f"\nGenerated {len(generated)} visual reports:")
            for name, path in generated.items():
                print(f"  - {name}: {path}")
        else:
            print("\nNo visual reports generated (matplotlib may not be available)")

    # Save JSON report
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPORTS_DIR / output_path

        try:
            report_dict = asdict(report)
            safe_write_json(str(output_path), report_dict, PROJECT_ROOT)
            logger.info(f"Report saved to {output_path}")
        except ValueError as e:
            logger.error(f"Could not save report: {e}")
            sys.exit(1)

    # Print summary
    print(f"\nAnalysis complete: {report.total_samples} samples analyzed")
    print(f"  Languages: {dict(report.language_distribution)}")
    print(f"  Exact duplicates: {report.exact_duplicates} groups")
    print(f"  Quality issues: {sum(report.quality_issues.values())} total")
    print(f"  Recommendations: {len(report.recommendations)}")


if __name__ == "__main__":
    main()
