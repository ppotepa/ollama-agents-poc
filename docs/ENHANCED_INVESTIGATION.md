# Enhanced Intelligent Investigation Mode

This document describes the improvements made to the intelligent investigation mode in the Ollama agent system.

## Overview

The enhanced intelligent investigation mode includes several key improvements:

1. **Better Prompt Recognition**: The system now uses pattern-based analysis to better understand the intent and complexity of user queries.

2. **Improved Model Selection**: More accurate task classification and model matching to ensure the right model is used for each task.

3. **Adaptive Learning**: The system learns from successful model selections and improves over time.

4. **Task Complexity Analysis**: Better handling of complex multi-step tasks.

## New Features

### Enhanced Task Classification

Tasks are now classified into more specific categories:
- CODE_GENERATION
- CODE_ANALYSIS
- CODE_OPTIMIZATION
- CODE_DEBUGGING
- FILE_OPERATIONS
- DEPENDENCY_MANAGEMENT
- ARCHITECTURE_DESIGN
- TESTING
- DOCUMENTATION
- SYSTEM_OPERATIONS
- RESEARCH
- GENERAL_QA

### Pattern-Based Prompt Recognition

The system now uses regex patterns to identify the true intent of a query, looking for patterns that indicate specific types of tasks.

### Complexity Assessment

Tasks are now assessed for complexity:
- SIMPLE: Basic tasks that require minimal context
- MEDIUM: Moderately complex tasks
- COMPLEX: Tasks requiring multiple steps or extensive context
- VERY_COMPLEX: Large, multi-component tasks

### Model Performance Tracking

The system now tracks which models perform best for which tasks and uses this information to improve future model selection.

## Usage

Enhanced investigation mode is enabled by default. To use it:

```bash
python main.py --mode intelligent --query "Your complex query here"
```

To disable enhanced investigation (fall back to the original implementation):

```bash
USE_ENHANCED_INVESTIGATION=0 python main.py --mode intelligent --query "Your complex query here"
```

## Configuration

The enhanced system can be configured through environment variables:

- `USE_ENHANCED_INVESTIGATION`: Set to "0" to disable enhanced features (default: "1")

## Examples

### Code Analysis

```bash
python main.py --mode intelligent --query "Analyze the performance bottlenecks in the file system operations"
```

### Complex Project Task

```bash
python main.py --mode intelligent --query "I need to create a new endpoint for user authentication that integrates with our existing database and supports both OAuth and basic authentication"
```

### Research Task

```bash
python main.py --mode intelligent --query "Research best practices for secure API implementation and summarize the key points"
```
