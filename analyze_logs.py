#!/usr/bin/env python3
"""Query Log Analysis Utility - Analyze and visualize query execution logs."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.query_analyzer import create_query_analyzer
from src.core.query_logger import get_query_logger
from src.utils.enhanced_logging import get_logger


def main():
    """Main entry point for log analysis utility."""
    parser = argparse.ArgumentParser(
        description="Analyze query execution logs and generate reports"
    )
    
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Run comprehensive analysis on recent logs"
    )
    
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        help="Number of days back to analyze (default: 7)"
    )
    
    parser.add_argument(
        "--export", "-e",
        type=str,
        help="Export detailed analytics to JSON file"
    )
    
    parser.add_argument(
        "--tree", "-t",
        type=str,
        help="Show execution tree visualization for specific query ID"
    )
    
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show summary of current session logs"
    )
    
    parser.add_argument(
        "--log-dir", "-l",
        type=str,
        default="logs/query_execution",
        help="Log directory path (default: logs/query_execution)"
    )
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_logger()
    
    if not any([args.analyze, args.export, args.tree, args.summary]):
        print("📊 Query Log Analysis Utility")
        print("=" * 40)
        print("Use --help to see available options")
        print("\nQuick start:")
        print("  python analyze_logs.py --analyze      # Analyze recent logs")
        print("  python analyze_logs.py --summary      # Show session summary")
        print("  python analyze_logs.py --export file  # Export detailed analytics")
        return
    
    # Create analyzer
    analyzer = create_query_analyzer(args.log_dir)
    
    if args.analyze:
        print(f"📊 Loading logs from last {args.days} days...")
        count = analyzer.load_logs(args.days)
        
        if count == 0:
            print("❌ No logs found to analyze")
            return
        
        print(f"✅ Loaded {count} query logs")
        print("\n" + "=" * 60)
        report = analyzer.generate_comprehensive_report()
        print(report)
    
    if args.export:
        print(f"📊 Loading logs for export...")
        count = analyzer.load_logs(args.days)
        
        if count == 0:
            print("❌ No logs found to export")
            return
        
        output_file = analyzer.export_detailed_analytics(args.export)
        if output_file:
            print(f"✅ Exported analytics to: {output_file}")
        else:
            print("❌ Failed to export analytics")
    
    if args.summary:
        query_logger = get_query_logger()
        summary = query_logger.get_session_summary()
        
        if "message" in summary:
            print(f"📊 {summary['message']}")
        else:
            session_data = summary["session_summary"]
            print("📊 SESSION SUMMARY")
            print("=" * 30)
            print(f"Total Queries: {session_data['total_queries']}")
            print(f"Success Rate: {session_data['success_rate']:.1%}")
            print(f"Average Time: {session_data['average_query_time']:.2f}s")
            print(f"Total Steps: {session_data['total_steps']}")
            print(f"Tools Executed: {session_data['total_tools_executed']}")
            
            if session_data['most_used_tools']:
                print("\nMost Used Tools:")
                for tool, count in session_data['most_used_tools']:
                    print(f"  🔧 {tool}: {count}")
            
            if session_data['most_used_models']:
                print("\nMost Used Models:")
                for model, count in session_data['most_used_models']:
                    print(f"  🤖 {model}: {count}")
    
    if args.tree:
        print(f"📊 Loading logs to find query {args.tree}...")
        count = analyzer.load_logs(args.days)
        
        tree_viz = analyzer.create_execution_tree_visualization(args.tree)
        print("\n" + "=" * 60)
        print(tree_viz)


if __name__ == "__main__":
    main()
