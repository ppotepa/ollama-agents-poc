# Intelligent Investigation System - Complete Implementation

## Overview

We have successfully implemented a comprehensive intelligent investigation system that enables dynamic model switching, task decomposition, execution planning, self-reflection, and adaptive investigation strategies. This system represents a significant advancement in automated AI agent orchestration.

## 🏗️ Architecture Components

### 1. Task Decomposer (`src/core/task_decomposer.py`)
**Purpose**: Break complex queries into manageable subtasks with optimal model assignments.

**Key Features**:
- **7 Task Types**: CODING, RESEARCH, FILE_ANALYSIS, SYSTEM_OPERATION, DATA_PROCESSING, CREATIVE, GENERAL_QA
- **Pattern Recognition**: Uses regex patterns to identify task characteristics
- **Model Preferences**: Automatically assigns optimal models based on task type
- **Complexity Estimation**: Provides complexity scores for resource planning
- **Dependency Management**: Tracks inter-task dependencies

**Example Output**:
```
Query: "Analyze the codebase structure and find any potential bugs"
→ 3 subtasks generated:
  1. Gather relevant context (FILE_ANALYSIS) → deepcoder:14b
  2. Scan directory structure (FILE_ANALYSIS) → deepcoder:14b  
  3. Analyze file content (FILE_ANALYSIS) → deepcoder:14b
```

### 2. Execution Planner (`src/core/execution_planner.py`)
**Purpose**: Create optimized execution plans with model assignments and dependency resolution.

**Key Features**:
- **Model Selection**: AgentResolver integration for optimal model matching
- **Dependency Resolution**: Ensures proper execution order
- **Performance Optimization**: Minimizes model switching overhead
- **Resource Management**: Tracks execution time and model transitions
- **Status Tracking**: PENDING → RUNNING → COMPLETED/FAILED workflow

**Capabilities**:
- Plans execution order considering dependencies
- Assigns models based on task requirements and availability
- Estimates total execution time
- Tracks model transition costs

### 3. Reflection System (`src/core/reflection_system.py`)
**Purpose**: Self-evaluation and automatic model swapping based on performance.

**Key Features**:
- **Confidence Assessment**: 5-level confidence scoring (VERY_LOW to VERY_HIGH)
- **Performance Metrics**: Accuracy, speed, reliability, user feedback
- **Automatic Triggers**: Step completion, errors, low confidence, timeouts
- **Model Recommendations**: Intelligent suggestions for model swaps
- **Improvement Suggestions**: Actionable recommendations for optimization

**Reflection Triggers**:
- After each step completion
- When errors are encountered
- When confidence drops below threshold
- On user request or timeout

### 4. Context Manager (`src/core/context_manager.py`)
**Purpose**: Persistent state management across model swaps and investigation sessions.

**Key Features**:
- **Session Management**: Persistent contexts with unique session IDs
- **State Transfer**: Seamless context transfer between model swaps
- **Execution History**: Complete audit trail of all operations
- **Data Persistence**: Optional file-based persistence for long sessions
- **Context Relevance**: Intelligent filtering of relevant context for each step

**Context Types**:
- Execution history and results
- Discovered files and analysis
- Error states and resolutions
- Progress tracking and metrics

### 5. Investigation Strategies (`src/core/investigation_strategies.py`)
**Purpose**: Different exploration patterns optimized for various query types.

**Implemented Strategies**:

#### **Depth-First Strategy**
- Dives deep into specific areas
- Ideal for: debugging, detailed analysis, focused implementation
- Creates specialized investigation steps for code, files, implementation, or debugging

#### **Breadth-First Strategy**
- Explores all areas broadly first
- Ideal for: overviews, comprehensive analysis, discovery
- Creates overview steps followed by area-specific investigations

#### **Targeted Strategy**
- Focuses on specific objectives
- Ideal for: specific files, functions, targeted fixes
- Extracts targets from queries and creates focused investigation plans

**Strategy Selection Logic**:
- "specific", "particular" → Targeted
- "overview", "summary", "all" → Breadth-First
- "deep", "detailed", "thorough" → Depth-First
- Default: Breadth-First

### 6. Intelligent Orchestrator (`src/core/intelligent_orchestrator.py`)
**Purpose**: Central coordination layer that integrates all components.

**Orchestration Process**:

1. **Planning Phase**:
   - Task decomposition using TaskDecomposer
   - Strategy selection via InvestigationStrategyManager
   - Execution plan creation combining investigation + decomposition

2. **Execution Phase**:
   - Step execution with assigned models
   - Real-time context management
   - Streaming or non-streaming execution modes

3. **Reflection Phase**:
   - Performance evaluation after each step
   - Model swap recommendations
   - Plan adaptation based on results

4. **Adaptation Phase**:
   - Dynamic plan modification
   - Strategy switching when needed
   - Context optimization

**Execution Modes**:
- **Sequential**: One step at a time with full reflection
- **Parallel**: Multiple independent steps concurrently
- **Adaptive**: Dynamic switching based on performance
- **Smart Routing**: Optimal model routing to minimize switches

## 🚀 Usage Examples

### 1. Basic Intelligent Investigation
```bash
python main.py --agent universal --intelligent -q "Analyze the codebase for optimization opportunities"
```

### 2. With Streaming Mode
```bash
python main.py --agent universal --intelligent --stream -q "Debug the authentication system"
```

### 3. Combined with Collaborative Mode
```bash
python main.py --agent universal --intelligent --collaborative -q "Implement a new user management feature"
```

## 🎯 Key Innovations

### 1. **Dynamic Model Switching**
- Automatic model selection based on task characteristics
- Performance-based model swapping during execution
- Context-aware model transitions

### 2. **Intelligent Task Decomposition**
- Pattern-based task identification
- Automatic complexity assessment
- Dependency-aware planning

### 3. **Self-Reflective Execution**
- Continuous performance monitoring
- Confidence-based decision making
- Adaptive strategy switching

### 4. **Persistent Context Management**
- State preservation across model swaps
- Intelligent context filtering
- Session-based investigation tracking

## 📊 Performance Optimizations

### 1. **Model Transition Minimization**
- Groups tasks by optimal model
- Reduces switching overhead
- Smart routing algorithms

### 2. **Parallel Execution Support**
- Independent task identification
- Concurrent execution capabilities
- Resource-aware scheduling

### 3. **Context Relevance Filtering**
- Intelligent context selection
- Memory usage optimization
- Performance-based pruning

## 🔧 Integration Points

### 1. **Streaming Mode Integration**
- Full compatibility with existing streaming infrastructure
- Real-time token output during investigation
- Non-blocking execution with progress updates

### 2. **Agent Factory Integration**
- Uses existing UniversalAgent infrastructure
- Leverages agent caching mechanisms
- Maintains streaming parameter propagation

### 3. **Model Configuration Integration**
- Works with existing models.yaml configuration
- Respects model swap tools settings
- Uses agent resolver for optimal selection

## 📈 Monitoring and Analytics

### 1. **Performance Metrics**
- Execution time tracking
- Model effectiveness scoring
- User satisfaction measurement

### 2. **Investigation Insights**
- Strategy performance analysis
- Model usage statistics
- Success rate tracking

### 3. **Session Statistics**
- Progress monitoring
- Resource utilization
- Error pattern analysis

## 🛠️ Configuration Options

### 1. **Orchestrator Settings**
- `max_session_duration`: Maximum investigation time
- `auto_reflection_interval`: Frequency of automatic reflection
- `confidence_threshold`: Minimum confidence for continuation
- `max_model_switches`: Limit on model changes per session

### 2. **Strategy Configuration**
- Custom investigation patterns
- Model preference overrides
- Execution mode defaults

### 3. **Context Management**
- Persistence directory configuration
- Context cleanup policies
- Memory usage limits

## 🎮 Command Line Interface

### New Flags Added:
- `--intelligent`: Enable intelligent investigation mode
- Integrates with existing flags:
  - `--stream`: Force streaming output
  - `--collaborative`: Enable collaborative mode
  - `--interception-mode`: Prompt analysis mode

### Example Commands:
```bash
# Basic intelligent investigation
python main.py --agent universal --intelligent -q "Find security vulnerabilities"

# Full-featured investigation
python main.py --agent universal --intelligent --collaborative --stream -q "Optimize database queries"

# Targeted investigation
python main.py --agent universal --intelligent -q "Fix the login bug in auth.py"
```

## 🧪 Testing and Validation

### Test Coverage:
- ✅ Task Decomposition functionality
- ✅ Execution Planning with model assignment
- ✅ Investigation Strategy selection
- ✅ Reflection System evaluation
- ✅ Context Manager state management
- ✅ Full system integration

### Validation Results:
- All core components working correctly
- Proper integration between systems
- Streaming mode compatibility verified
- Model switching logic functional

## 🔮 Future Enhancements

### 1. **Advanced Strategy Patterns**
- Hypothesis-driven investigation
- Incremental exploration strategies
- User-defined investigation patterns

### 2. **Machine Learning Integration**
- Performance-based model learning
- Strategy effectiveness optimization
- Automatic pattern recognition improvement

### 3. **Enhanced Parallel Execution**
- Advanced dependency resolution
- Resource-aware task scheduling
- Dynamic load balancing

### 4. **Extended Context Management**
- Cross-session context sharing
- Long-term memory systems
- Knowledge base integration

## 📝 Summary

This intelligent investigation system represents a complete overhaul of how AI agents can work together to solve complex problems. By combining task decomposition, intelligent execution planning, self-reflection, and adaptive strategies, we've created a system that can:

1. **Automatically break down complex queries** into manageable subtasks
2. **Select optimal models** for each specific task type
3. **Execute investigations** using the most appropriate strategy
4. **Monitor performance** and adapt in real-time
5. **Maintain context** across model switches and execution phases
6. **Provide streaming output** with real-time progress updates

The system is fully integrated with the existing codebase and provides a powerful new capability for handling complex, multi-step investigations that require different specialized models working together intelligently.

**To use the system**: `python main.py --agent universal --intelligent -q 'your complex query here'`
