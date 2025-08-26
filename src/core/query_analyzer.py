"""Query Analysis and Visualization Tools - Analyze logged query execution data."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.utils.enhanced_logging import get_logger


@dataclass
class QueryAnalytics:
    """Analytics data for query execution patterns."""
    total_queries: int
    success_rate: float
    average_execution_time: float
    most_used_tools: list[tuple[str, int]]
    most_used_models: list[tuple[str, int]]
    context_usage_patterns: dict[str, Any]
    execution_patterns: dict[str, Any]
    error_patterns: list[str]


class QueryAnalyzer:
    """Analyzer for logged query execution data."""

    def __init__(self, log_directory: str = "logs/query_execution"):
        """Initialize the query analyzer."""
        self.log_directory = Path(log_directory)
        self.logger = get_logger()
        self.loaded_logs: list[dict[str, Any]] = []

    def load_logs(self, days_back: int = 7) -> int:
        """Load logs from the specified number of days back."""
        if not self.log_directory.exists():
            self.logger.warning(f"Log directory {self.log_directory} does not exist")
            return 0

        cutoff_date = datetime.now() - timedelta(days=days_back)
        loaded_count = 0

        for log_file in self.log_directory.glob("query_log_*.json"):
            try:
                # Extract timestamp from filename
                timestamp_str = log_file.stem.split('_')[2]  # query_log_TIMESTAMP_ID.json
                file_date = datetime.strptime(timestamp_str, "%Y%m%d")

                if file_date >= cutoff_date:
                    with open(log_file, encoding='utf-8') as f:
                        log_data = json.load(f)
                        self.loaded_logs.append(log_data)
                        loaded_count += 1

            except Exception as e:
                self.logger.warning(f"Failed to load log file {log_file}: {e}")

        self.logger.info(f"📊 Loaded {loaded_count} query logs from last {days_back} days")
        return loaded_count

    def analyze_execution_patterns(self) -> dict[str, Any]:
        """Analyze execution patterns across loaded logs."""
        if not self.loaded_logs:
            return {"error": "No logs loaded"}

        patterns = {
            "execution_modes": Counter(),
            "step_distributions": [],
            "tool_sequences": [],
            "agent_switch_patterns": [],
            "context_evolution": {},
            "prompt_decoration_patterns": Counter()
        }

        for log in self.loaded_logs:
            # Execution modes
            patterns["execution_modes"][log.get("execution_mode", "unknown")] += 1

            # Step distributions
            step_count = len(log.get("execution_steps", []))
            patterns["step_distributions"].append(step_count)

            # Tool sequences
            tools_used = []
            for step in log.get("execution_steps", []):
                step_tools = [tool["tool_name"] for tool in step.get("tools_executed", [])]
                tools_used.extend(step_tools)
            if tools_used:
                patterns["tool_sequences"].append(tools_used)

            # Agent switch patterns
            switches = log.get("agent_switches", [])
            if switches:
                switch_sequence = []
                for switch in switches:
                    switch_sequence.append({
                        "from": switch.get("from_agent"),
                        "to": switch.get("to_agent"),
                        "reason": switch.get("reason", "")
                    })
                patterns["agent_switch_patterns"].append(switch_sequence)

            # Prompt decorations
            for decoration in log.get("prompt_decorations", []):
                for decoration_type in decoration.get("decorations_applied", []):
                    patterns["prompt_decoration_patterns"][decoration_type] += 1

        return patterns

    def analyze_context_usage(self) -> dict[str, Any]:
        """Analyze how context is used across queries."""
        if not self.loaded_logs:
            return {"error": "No logs loaded"}

        context_analysis = {
            "sources": Counter(),
            "size_distributions": [],
            "evolution_patterns": {},
            "effectiveness_metrics": {}
        }

        for log in self.loaded_logs:
            # Context sources
            for context in log.get("total_context_used", []):
                context_analysis["sources"][context.get("source", "unknown")] += 1
                context_analysis["size_distributions"].append(context.get("size_chars", 0))

            # Context evolution
            evolution = log.get("context_evolution", [])
            if evolution:
                context_analysis["evolution_patterns"][log["query_id"]] = {
                    "steps": len(evolution),
                    "context_growth": [e.get("context_state", {}) for e in evolution]
                }

        # Calculate statistics
        if context_analysis["size_distributions"]:
            sizes = context_analysis["size_distributions"]
            context_analysis["effectiveness_metrics"] = {
                "average_context_size": sum(sizes) / len(sizes),
                "max_context_size": max(sizes),
                "min_context_size": min(sizes),
                "total_context_chars": sum(sizes)
            }

        return context_analysis

    def analyze_tool_effectiveness(self) -> dict[str, Any]:
        """Analyze tool usage and effectiveness."""
        if not self.loaded_logs:
            return {"error": "No logs loaded"}

        tool_analysis = {
            "usage_frequency": Counter(),
            "success_rates": {},
            "execution_times": defaultdict(list),
            "follow_up_patterns": defaultdict(list),
            "output_sizes": defaultdict(list)
        }

        for log in self.loaded_logs:
            for tool_exec in log.get("tools_executed", []):
                tool_name = tool_exec.get("tool_name", "unknown")

                # Usage frequency
                tool_analysis["usage_frequency"][tool_name] += 1

                # Success tracking
                if tool_name not in tool_analysis["success_rates"]:
                    tool_analysis["success_rates"][tool_name] = {"total": 0, "successful": 0}

                tool_analysis["success_rates"][tool_name]["total"] += 1
                if tool_exec.get("success", False):
                    tool_analysis["success_rates"][tool_name]["successful"] += 1

                # Performance metrics
                tool_analysis["execution_times"][tool_name].append(tool_exec.get("execution_time", 0))
                tool_analysis["output_sizes"][tool_name].append(tool_exec.get("output_size", 0))

                # Follow-up patterns
                follow_ups = tool_exec.get("follow_up_tools", [])
                if follow_ups:
                    tool_analysis["follow_up_patterns"][tool_name].extend(follow_ups)

        # Calculate success rates and averages
        for tool_name, data in tool_analysis["success_rates"].items():
            data["success_rate"] = data["successful"] / data["total"] if data["total"] > 0 else 0

        return tool_analysis

    def analyze_model_performance(self) -> dict[str, Any]:
        """Analyze model/agent performance patterns."""
        if not self.loaded_logs:
            return {"error": "No logs loaded"}

        model_analysis = {
            "usage_frequency": Counter(),
            "success_rates": {},
            "execution_times": defaultdict(list),
            "switch_reasons": defaultdict(list),
            "task_type_preferences": defaultdict(Counter)
        }

        for log in self.loaded_logs:
            # Model usage in steps
            for step in log.get("execution_steps", []):
                agent_used = step.get("agent_used", "unknown")
                model_analysis["usage_frequency"][agent_used] += 1

                # Track execution times per model
                exec_time = step.get("execution_time", 0)
                if exec_time > 0:
                    model_analysis["execution_times"][agent_used].append(exec_time)

            # Agent switches
            for switch in log.get("agent_switches", []):
                to_agent = switch.get("to_agent", "unknown")
                reason = switch.get("reason", "unknown")
                model_analysis["switch_reasons"][to_agent].append(reason)

            # Success rates per model
            query_success = log.get("success", False)
            models_used = log.get("models_used", [])
            for model in models_used:
                if model not in model_analysis["success_rates"]:
                    model_analysis["success_rates"][model] = {"total": 0, "successful": 0}

                model_analysis["success_rates"][model]["total"] += 1
                if query_success:
                    model_analysis["success_rates"][model]["successful"] += 1

        # Calculate success rates
        for model, data in model_analysis["success_rates"].items():
            data["success_rate"] = data["successful"] / data["total"] if data["total"] > 0 else 0

        return model_analysis

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        if not self.loaded_logs:
            return "📊 No query logs available for analysis."

        # Run all analyses
        execution_patterns = self.analyze_execution_patterns()
        context_usage = self.analyze_context_usage()
        tool_effectiveness = self.analyze_tool_effectiveness()
        model_performance = self.analyze_model_performance()

        # Calculate overall metrics
        total_queries = len(self.loaded_logs)
        successful_queries = sum(1 for log in self.loaded_logs if log.get("success", False))
        success_rate = successful_queries / total_queries if total_queries > 0 else 0

        total_time = sum(log.get("total_execution_time", 0) for log in self.loaded_logs)
        avg_time = total_time / total_queries if total_queries > 0 else 0

        # Build report
        report = []
        report.append("📊 COMPREHENSIVE QUERY EXECUTION ANALYSIS")
        report.append("=" * 60)
        report.append("")

        # Overall metrics
        report.append("🎯 OVERALL METRICS")
        report.append("-" * 30)
        report.append(f"📈 Total Queries Analyzed: {total_queries}")
        report.append(f"✅ Success Rate: {success_rate:.1%}")
        report.append(f"⏱️  Average Execution Time: {avg_time:.2f}s")
        report.append(f"🕐 Total Execution Time: {total_time:.2f}s")
        report.append("")

        # Execution patterns
        report.append("🔄 EXECUTION PATTERNS")
        report.append("-" * 30)
        mode_counter = execution_patterns["execution_modes"]
        for mode, count in mode_counter.most_common():
            report.append(f"📋 {mode}: {count} queries ({count/total_queries:.1%})")

        step_dist = execution_patterns["step_distributions"]
        if step_dist:
            avg_steps = sum(step_dist) / len(step_dist)
            report.append(f"🔢 Average Steps per Query: {avg_steps:.1f}")
            report.append(f"📊 Step Range: {min(step_dist)}-{max(step_dist)}")
        report.append("")

        # Tool effectiveness
        report.append("🔧 TOOL EFFECTIVENESS")
        report.append("-" * 30)
        tool_freq = tool_effectiveness["usage_frequency"]
        for tool, count in tool_freq.most_common(10):
            success_data = tool_effectiveness["success_rates"].get(tool, {})
            success_rate = success_data.get("success_rate", 0)
            avg_time = 0
            if tool in tool_effectiveness["execution_times"]:
                times = tool_effectiveness["execution_times"][tool]
                avg_time = sum(times) / len(times) if times else 0

            report.append(f"🛠️  {tool}: {count} uses, {success_rate:.1%} success, {avg_time:.3f}s avg")
        report.append("")

        # Model performance
        report.append("🤖 MODEL PERFORMANCE")
        report.append("-" * 30)
        model_freq = model_performance["usage_frequency"]
        for model, count in model_freq.most_common(10):
            success_data = model_performance["success_rates"].get(model, {})
            success_rate = success_data.get("success_rate", 0)
            avg_time = 0
            if model in model_performance["execution_times"]:
                times = model_performance["execution_times"][model]
                avg_time = sum(times) / len(times) if times else 0

            report.append(f"🧠 {model}: {count} steps, {success_rate:.1%} success, {avg_time:.3f}s avg")
        report.append("")

        # Context usage
        report.append("📚 CONTEXT USAGE")
        report.append("-" * 30)
        context_sources = context_usage["sources"]
        for source, count in context_sources.most_common():
            report.append(f"📄 {source}: {count} instances")

        metrics = context_usage.get("effectiveness_metrics", {})
        if metrics:
            report.append(f"📏 Average Context Size: {metrics.get('average_context_size', 0):.0f} chars")
            report.append(f"📊 Total Context Used: {metrics.get('total_context_chars', 0):,} chars")
        report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 30)

        # Tool recommendations
        tool_success_rates = {tool: data.get("success_rate", 0)
                            for tool, data in tool_effectiveness["success_rates"].items()}
        low_success_tools = [tool for tool, rate in tool_success_rates.items() if rate < 0.8]
        if low_success_tools:
            report.append(f"⚠️  Tools with low success rates: {', '.join(low_success_tools)}")

        # Model recommendations
        model_success_rates = {model: data.get("success_rate", 0)
                             for model, data in model_performance["success_rates"].items()}
        high_success_models = [model for model, rate in model_success_rates.items() if rate > 0.9]
        if high_success_models:
            report.append(f"⭐ High-performing models: {', '.join(high_success_models)}")

        if success_rate < 0.8:
            report.append("🔍 Consider investigating failure patterns to improve success rate")

        if avg_time > 30:
            report.append("⏱️  Consider optimizing execution flow to reduce average time")

        return "\n".join(report)

    def export_detailed_analytics(self, output_file: str = None) -> str:
        """Export detailed analytics to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"query_analytics_{timestamp}.json"

        analytics_data = {
            "generation_time": datetime.now().isoformat(),
            "total_logs_analyzed": len(self.loaded_logs),
            "execution_patterns": self.analyze_execution_patterns(),
            "context_usage": self.analyze_context_usage(),
            "tool_effectiveness": self.analyze_tool_effectiveness(),
            "model_performance": self.analyze_model_performance(),
            "summary_metrics": {
                "total_queries": len(self.loaded_logs),
                "success_rate": sum(1 for log in self.loaded_logs if log.get("success", False)) / len(self.loaded_logs) if self.loaded_logs else 0,
                "average_execution_time": sum(log.get("total_execution_time", 0) for log in self.loaded_logs) / len(self.loaded_logs) if self.loaded_logs else 0
            }
        }

        # Convert Counter objects to regular dicts for JSON serialization
        def convert_counters(obj):
            if isinstance(obj, Counter):
                return dict(obj)
            elif isinstance(obj, dict):
                return {k: convert_counters(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_counters(item) for item in obj]
            else:
                return obj

        analytics_data = convert_counters(analytics_data)

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analytics_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"📊 Exported detailed analytics to: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"❌ Failed to export analytics: {e}")
            return ""


def create_query_analyzer(log_directory: str = "logs/query_execution") -> QueryAnalyzer:
    """Create and initialize a query analyzer."""
    return QueryAnalyzer(log_directory)


__all__ = ["QueryAnalyzer", "QueryAnalytics", "create_query_analyzer"]
