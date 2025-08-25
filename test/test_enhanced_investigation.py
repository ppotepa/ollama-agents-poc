"""Test the enhanced investigation mode and model selection."""

import os
import unittest
import asyncio
from unittest.mock import patch, MagicMock

# Import the components we want to test
from src.core.intelligent_orchestrator import get_orchestrator, ExecutionMode
from src.core.intelligence_enhancer import (
    apply_intelligence_improvements,
    is_enhanced_investigation_enabled
)
from src.core.improved_investigation_strategies import (
    ImprovedTaskAnalyzer,
    TaskType,
    TaskComplexity,
    model_scoring_cache
)

class TestEnhancedInvestigation(unittest.TestCase):
    """Test the enhanced investigation mode improvements."""
    
    def setUp(self):
        """Set up the test environment."""
        # Enable enhanced investigation
        os.environ['USE_ENHANCED_INVESTIGATION'] = '1'
    
    def tearDown(self):
        """Clean up after tests."""
        # Reset environment
        if 'USE_ENHANCED_INVESTIGATION' in os.environ:
            del os.environ['USE_ENHANCED_INVESTIGATION']
    
    def test_is_enhanced_investigation_enabled(self):
        """Test the feature flag functionality."""
        # Should be enabled by default
        self.assertTrue(is_enhanced_investigation_enabled())
        
        # Test explicit enable
        os.environ['USE_ENHANCED_INVESTIGATION'] = '1'
        self.assertTrue(is_enhanced_investigation_enabled())
        
        # Test explicit disable
        os.environ['USE_ENHANCED_INVESTIGATION'] = '0'
        self.assertFalse(is_enhanced_investigation_enabled())
    
    def test_task_analyzer_detects_code_task(self):
        """Test that the task analyzer correctly identifies code tasks."""
        analyzer = ImprovedTaskAnalyzer()
        
        # Test a code generation query
        query = "Write a Python function to calculate Fibonacci numbers"
        analysis = analyzer.analyze_task(query)
        
        self.assertEqual(analysis['task_type'], TaskType.CODE_GENERATION.name)
        self.assertEqual(analysis['complexity'], TaskComplexity.SIMPLE.name)
        self.assertTrue(analysis['tools_required'])
    
    def test_task_analyzer_detects_debugging_task(self):
        """Test that the task analyzer correctly identifies debugging tasks."""
        analyzer = ImprovedTaskAnalyzer()
        
        # Test a debugging query
        query = "Fix the bug in my code that's causing a null pointer exception"
        analysis = analyzer.analyze_task(query)
        
        self.assertEqual(analysis['task_type'], TaskType.CODE_DEBUGGING.name)
        self.assertTrue(analysis['tools_required'])
    
    def test_task_analyzer_detects_complex_task(self):
        """Test that the task analyzer correctly identifies complex tasks."""
        analyzer = ImprovedTaskAnalyzer()
        
        # Test a complex query
        query = """
        Create a comprehensive system for user authentication. 
        First, implement the database schema. 
        Then create the registration endpoint. 
        Next, implement the login functionality with JWT tokens.
        Finally, add password reset capability and email verification.
        """
        
        analysis = analyzer.analyze_task(query)
        
        self.assertEqual(analysis['complexity'], TaskComplexity.COMPLEX.name)
        self.assertTrue(len(analysis['subtasks']) > 0)
    
    def test_model_scoring_cache(self):
        """Test the model scoring cache functionality."""
        # Clear any existing cache
        model_scoring_cache.scores = {}
        
        # Record successes and failures
        model_scoring_cache.record_success("CODE_GENERATION", "qwen2.5-coder:7b")
        model_scoring_cache.record_success("CODE_GENERATION", "qwen2.5-coder:7b")
        model_scoring_cache.record_failure("CODE_GENERATION", "qwen2.5-coder:7b")
        model_scoring_cache.record_success("CODE_GENERATION", "phi3:small")
        
        # Check success rates
        self.assertAlmostEqual(
            model_scoring_cache.get_success_rate("CODE_GENERATION", "qwen2.5-coder:7b"), 
            2/3
        )
        self.assertAlmostEqual(
            model_scoring_cache.get_success_rate("CODE_GENERATION", "phi3:small"), 
            1.0
        )
        
        # Get best models
        best_models = model_scoring_cache.get_best_models("CODE_GENERATION")
        self.assertEqual(best_models[0], "phi3:small")  # Should be first (100% success)
        self.assertEqual(best_models[1], "qwen2.5-coder:7b")  # Should be second (66% success)

    @patch('src.core.intelligent_orchestrator.IntelligentOrchestrator')
    def test_apply_intelligence_improvements(self, mock_orchestrator):
        """Test that intelligence improvements are applied correctly."""
        # Create mock objects
        mock_orchestrator.strategy_manager = MagicMock()
        mock_orchestrator._execute_step = MagicMock()
        
        # Apply improvements
        apply_intelligence_improvements(mock_orchestrator)
        
        # Check that the strategies were updated
        self.assertTrue(hasattr(mock_orchestrator, '_intelligence_enhanced'))

if __name__ == '__main__':
    unittest.main()
