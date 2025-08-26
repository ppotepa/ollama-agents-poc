"""
Prompt Interceptor Module - Legacy Interface

This module provides backwards compatibility for the intelligent prompt interception system.
The core functionality has been refactored into modular components under prompt_analysis/
"""

# Import from the new modular system
from .prompt_analysis import (
    PromptEnhancer, 
    InterceptionMode, 
    CommandExecution, 
    SupplementedPrompt,
    PromptType,
    PromptContext
)

class PromptInterceptor:
    def __init__(self):
        self.enhancer = PromptEnhancer()
    
    def intercept_and_enhance(self, prompt: str, working_directory: str = None, 
                             mode: InterceptionMode = InterceptionMode.SMART) -> SupplementedPrompt:
        return self.enhancer.enhance_prompt(prompt, working_directory, mode)

__all__ = ['PromptInterceptor', 'InterceptionMode', 'PromptType', 'PromptContext', 'CommandExecution', 'SupplementedPrompt']
