# Comprehensive Query Logging System

## Overview
We have implemented a comprehensive query logging system that captures detailed information about query execution, context usage, prompt decorations, agent switches, tool executions, and execution trees.

## Components

### 1. Core Logging System (`src/core/query_logger.py`)

#### Data Structures
- **`QueryExecutionLog`**: Complete log of a query execution session
- **`ExecutionStep`**: Individual steps in the execution process
- **`ContextInfo`**: Information about context used during execution
- **`PromptDecoration`**: Details about how prompts were enhanced
- **`AgentSwitch`**: Information about model/agent switching
- **`ToolExecution`**: Details about tool usage and results

#### Key Features
- **Automatic Session Management**: Each query gets a unique ID and comprehensive tracking
- **Context Evolution Tracking**: Monitors how context grows and changes during execution
- **Prompt Decoration Logging**: Captures how original prompts are enhanced with system messages and context
- **Tool Execution Metrics**: Tracks tool usage, success rates, execution times, and output sizes
- **Agent Switch Tracking**: Logs model switches with reasons and context preservation details

### 2. Integration Layer (`src/core/query_logger_integration.py`)

#### Wrapper Classes
- **`LoggingCollaborativeWrapper`**: Wraps collaborative system with logging
- **`LoggingUniversalAgentWrapper`**: Wraps Universal Multi-Agent with logging

#### Features
- **Non-invasive Integration**: Patches existing systems without modifying core code
- **Automatic Logging**: Captures execution details transparently
- **Error Handling**: Gracefully handles logging failures without breaking execution

### 3. Analysis System (`src/core/query_analyzer.py`)

#### Analytics Capabilities
- **Execution Pattern Analysis**: Identifies common execution flows and patterns
- **Context Usage Analysis**: Analyzes how context is used and its effectiveness
- **Tool Effectiveness Analysis**: Measures tool success rates and performance
- **Model Performance Analysis**: Compares model performance across different tasks

#### Report Generation
- **Comprehensive Reports**: Detailed analysis with metrics and recommendations
- **Export Functionality**: JSON export for further analysis
- **Visualization**: Text-based execution tree visualization

## What Gets Logged

### 1. Query Information
```json
{
  "query_id": "2e700cb7",
  "original_query": "test query - what is 2+2?",
  "execution_mode": "collaborative",
  "start_time": "2025-08-26T15:10:54.027234",
  "end_time": "2025-08-26T15:10:54.845123",
  "total_execution_time": 0.818
}
```

### 2. Context Usage
```json
{
  "source": "user_input",
  "content": "test query - what is 2+2?",
  "size_chars": 25,
  "timestamp": "2025-08-26T15:10:54.027234",
  "metadata": {
    "type": "math_question",
    "complexity": "simple"
  }
}
```

### 3. Prompt Decorations
```json
{
  "original_prompt": "what is 2+2?",
  "decorated_prompt": "You are a helpful assistant. Please answer: what is 2+2?",
  "decorations_applied": ["system_message", "instruction_prefix"],
  "context_added": 42,
  "system_message": "You are a helpful assistant.",
  "prompt_hash": "a1b2c3d4e5f6"
}
```

### 4. Tool Executions
```json
{
  "tool_name": "calculator",
  "args": {"expression": "2+2"},
  "execution_time": 0.001,
  "success": true,
  "output_size": 1,
  "output_preview": "4",
  "follow_up_tools": ["verify_calculation"],
  "timestamp": "2025-08-26T15:10:54.027234"
}
```

### 5. Agent Switches
```json
{
  "from_agent": "qwen2.5:7b-instruct-q4_K_M",
  "to_agent": "deepcoder:14b",
  "reason": "Better for coding tasks",
  "timestamp": "2025-08-26T15:10:54.500000",
  "context_preserved": "Previous analysis and discovered files...",
  "context_size": 1018,
  "success": true
}
```

### 6. Execution Steps
```json
{
  "step_number": 1,
  "step_type": "collaborative_step",
  "agent_used": "qwen2.5:7b-instruct-q4_K_M",
  "description": "Analyzing user query and selecting tools",
  "tools_executed": [...],
  "agent_switches": [...],
  "context_used": [...],
  "prompt_decorations": [...],
  "output_generated": "Analysis complete...",
  "execution_time": 2.45
}
```

## Usage Examples

### Manual Logging
```python
from src.core.query_logger import get_query_logger

# Get logger instance
query_logger = get_query_logger()

# Start session
query_id = query_logger.start_query_session("What is machine learning?", "educational")

# Log execution step
step_id = query_logger.start_execution_step(
    "research", 
    "qwen2.5:7b-instruct-q4_K_M", 
    "Researching machine learning concepts"
)

# Log context usage
query_logger.log_context_usage(
    source="knowledge_base",
    content="Machine learning is a subset of artificial intelligence...",
    metadata={"domain": "computer_science", "confidence": 0.95}
)

# Log tool execution
query_logger.log_tool_execution(
    tool_name="search_documents",
    args={"query": "machine learning definition"},
    execution_time=0.5,
    success=True,
    output="Found 15 relevant documents"
)

# End step and session
query_logger.end_execution_step("Research completed successfully")
query_logger.end_query_session("Machine learning is...", success=True)
```

### Analysis and Reporting
```python
from src.core.query_analyzer import create_query_analyzer

# Create analyzer
analyzer = create_query_analyzer()

# Load recent logs
analyzer.load_logs(days_back=7)

# Generate comprehensive report
report = analyzer.generate_comprehensive_report()
print(report)

# Export detailed analytics
analyzer.export_detailed_analytics("weekly_analytics.json")
```

### Command Line Analysis
```bash
# Analyze recent logs
python analyze_logs.py --analyze --days 7

# Show session summary
python analyze_logs.py --summary

# Export detailed analytics
python analyze_logs.py --export weekly_report.json --days 7

# Show execution tree for specific query
python analyze_logs.py --tree 2e700cb7
```

## Analytics and Insights

### Execution Patterns
- **Mode Distribution**: Which execution modes are used most frequently
- **Step Patterns**: Average steps per query, step distribution ranges
- **Tool Sequences**: Common sequences of tools used together
- **Agent Switch Patterns**: When and why agents are switched

### Context Analysis
- **Source Distribution**: Where context comes from (user input, files, tools, etc.)
- **Size Patterns**: How much context is typically used
- **Evolution Tracking**: How context grows during execution
- **Effectiveness Metrics**: Which context sources are most valuable

### Tool Effectiveness
- **Usage Frequency**: Most commonly used tools
- **Success Rates**: Which tools are most reliable
- **Performance Metrics**: Execution times and output sizes
- **Follow-up Patterns**: Which tools are often used together

### Model Performance
- **Usage Distribution**: Which models are used most frequently
- **Success Rates**: Model reliability across different task types
- **Switch Patterns**: Why and when models are switched
- **Task Preferences**: Which models work best for specific task types

## Benefits

### 1. **Debugging and Troubleshooting**
- Detailed execution traces help identify where things go wrong
- Context usage analysis reveals missing or ineffective context
- Tool execution logs show which operations are failing

### 2. **Performance Optimization**
- Identify slow tools and optimize them
- Analyze model switch patterns to improve agent selection
- Monitor context usage to optimize prompt construction

### 3. **System Understanding**
- Visualize execution flows with tree representations
- Understand how context evolves during complex queries
- Track the effectiveness of different execution strategies

### 4. **Quality Assurance**
- Monitor success rates across different query types
- Identify patterns in failures for systematic improvements
- Track system performance over time

### 5. **User Experience Insights**
- Understand which types of queries work best
- Identify common failure points for user guidance
- Optimize system behavior based on usage patterns

## File Locations

- **Logs**: `logs/query_execution/query_log_TIMESTAMP_ID.json`
- **Analytics**: Exported to specified files (e.g., `weekly_analytics.json`)
- **Configuration**: Logging directory configurable in QueryLogger constructor

## Integration Status

✅ **Core Logging System**: Fully implemented and tested
✅ **Analysis Tools**: Comprehensive analytics and reporting
✅ **Export Functionality**: JSON export for detailed analysis
✅ **Command Line Tools**: Easy-to-use analysis utilities
✅ **Visualization**: Text-based execution tree visualization
🔄 **System Integration**: Requires fixing import issues for automatic patching

The logging system provides unprecedented visibility into query execution, enabling data-driven improvements to the collaborative agent system.
