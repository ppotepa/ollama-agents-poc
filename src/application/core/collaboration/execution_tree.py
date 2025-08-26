"""Execution tree components for tracking collaborative execution flow."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional


class ExecutionNodeType(Enum):
    """Types of nodes in the execution tree."""
    USER_QUERY = "user_query"
    INTERCEPTOR_ANALYSIS = "interceptor_analysis"
    COMMAND_EXECUTION = "command_execution"
    AGENT_RESPONSE = "agent_response"
    FOLLOW_UP_QUERY = "follow_up_query"
    FINAL_RESULT = "final_result"


@dataclass
class ExecutionNode:
    """A node in the execution tree representing a step in the collaborative process."""
    node_type: ExecutionNodeType
    content: str
    metadata: dict[str, Any]
    children: List['ExecutionNode']
    timestamp: float
    execution_time: float = 0.0
    success: bool = True
    parent: Optional['ExecutionNode'] = None

    def add_child(self, child: 'ExecutionNode') -> 'ExecutionNode':
        """Add a child node and set parent reference."""
        child.parent = self
        self.children.append(child)
        return child

    def get_depth(self) -> int:
        """Get the depth of this node in the tree."""
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    def get_path_to_root(self) -> List['ExecutionNode']:
        """Get the path from this node to the root."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def find_nodes_by_type(self, node_type: ExecutionNodeType) -> List['ExecutionNode']:
        """Find all descendant nodes of a specific type."""
        results = []
        if self.node_type == node_type:
            results.append(self)
        
        for child in self.children:
            results.extend(child.find_nodes_by_type(node_type))
        
        return results

    def get_execution_summary(self) -> dict[str, Any]:
        """Get a summary of this node's execution."""
        return {
            "node_type": self.node_type.value,
            "content_preview": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "execution_time": self.execution_time,
            "success": self.success,
            "timestamp": self.timestamp,
            "children_count": len(self.children),
            "metadata_keys": list(self.metadata.keys())
        }


class ExecutionTreeManager:
    """Manager for execution tree operations and analysis."""

    def __init__(self, root_node: ExecutionNode):
        """Initialize with a root node."""
        self.root = root_node

    def create_initial_tree(self, query: str, working_directory: str) -> ExecutionNode:
        """Create the initial execution tree for a query."""
        return ExecutionNode(
            node_type=ExecutionNodeType.USER_QUERY,
            content=query,
            metadata={"query": query, "working_directory": working_directory},
            children=[],
            timestamp=time.time()
        )

    def get_total_execution_time(self) -> float:
        """Get the total execution time for the entire tree."""
        def sum_time(node: ExecutionNode) -> float:
            total = node.execution_time
            for child in node.children:
                total += sum_time(child)
            return total
        
        return sum_time(self.root)

    def get_execution_timeline(self) -> List[dict[str, Any]]:
        """Get a chronological timeline of all execution steps."""
        timeline = []
        
        def collect_nodes(node: ExecutionNode):
            timeline.append({
                "timestamp": node.timestamp,
                "node_type": node.node_type.value,
                "content": node.content,
                "execution_time": node.execution_time,
                "success": node.success,
                "depth": node.get_depth()
            })
            
            for child in sorted(node.children, key=lambda x: x.timestamp):
                collect_nodes(child)
        
        collect_nodes(self.root)
        return sorted(timeline, key=lambda x: x["timestamp"])

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the execution tree."""
        all_nodes = []
        
        def collect_all_nodes(node: ExecutionNode):
            all_nodes.append(node)
            for child in node.children:
                collect_all_nodes(child)
        
        collect_all_nodes(self.root)
        
        if not all_nodes:
            return {}
        
        execution_times = [node.execution_time for node in all_nodes if node.execution_time > 0]
        successful_nodes = [node for node in all_nodes if node.success]
        
        return {
            "total_nodes": len(all_nodes),
            "successful_nodes": len(successful_nodes),
            "success_rate": len(successful_nodes) / len(all_nodes) if all_nodes else 0,
            "total_execution_time": sum(execution_times),
            "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "min_execution_time": min(execution_times) if execution_times else 0,
            "node_type_distribution": self._get_node_type_distribution(all_nodes)
        }

    def _get_node_type_distribution(self, nodes: List[ExecutionNode]) -> dict[str, int]:
        """Get the distribution of node types in the tree."""
        distribution = {}
        for node in nodes:
            node_type = node.node_type.value
            distribution[node_type] = distribution.get(node_type, 0) + 1
        return distribution

    def export_tree_structure(self) -> dict[str, Any]:
        """Export the complete tree structure."""
        def serialize_node(node: ExecutionNode) -> dict[str, Any]:
            return {
                "node_type": node.node_type.value,
                "content": node.content,
                "metadata": node.metadata,
                "timestamp": node.timestamp,
                "execution_time": node.execution_time,
                "success": node.success,
                "children": [serialize_node(child) for child in node.children]
            }
        
        return {
            "root": serialize_node(self.root),
            "metrics": self.get_performance_metrics(),
            "timeline": self.get_execution_timeline()
        }

    def find_critical_path(self) -> List[ExecutionNode]:
        """Find the path with the longest execution time."""
        def get_path_time(node: ExecutionNode) -> tuple[float, List[ExecutionNode]]:
            if not node.children:
                return node.execution_time, [node]
            
            max_time = 0
            max_path = []
            
            for child in node.children:
                child_time, child_path = get_path_time(child)
                total_time = node.execution_time + child_time
                
                if total_time > max_time:
                    max_time = total_time
                    max_path = [node] + child_path
            
            return max_time, max_path
        
        _, critical_path = get_path_time(self.root)
        return critical_path
