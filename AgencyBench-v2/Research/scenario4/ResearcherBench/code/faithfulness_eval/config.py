"""
Simplified configuration management for DAIR Benchmark
"""
import os
from typing import Optional


class Config:
    """Simplified configuration class"""
    
    def __init__(self):
        """
        Initialize configuration with environment variables
        """
        # API Configuration - try environment variables first
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("MODEL")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # External API keys
        self.jina_api_key = os.getenv("JINA_API_KEY", "")
        
        # Simple parameters with sensible defaults
        self.max_workers = int(os.getenv("MAX_WORKERS", "5"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.claims_step_size = int(os.getenv("CLAIMS_STEP_SIZE", "5"))
        self.max_content_length = int(os.getenv("MAX_CONTENT_LENGTH", "950000"))  # Max chars for web content
        
        # Directory paths
        self.output_dir = os.getenv("OUTPUT_DIR", "./results")
        self.claims_dir = os.getenv("CLAIMS_DIR", "./results/claims")
        self.logs_dir = os.getenv("LOGS_DIR", "./results/logs")
    
    def validate(self) -> bool:
        """Simple validation"""
        if not self.api_key:
            print("❌ Error: No API key found. Please set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.")
            return False
        
        return True
    
    def show_config(self):
        """Display current configuration (without showing API keys)"""
        print("🔧 Current Configuration:")
        print(f"   Model: {self.model}")
        print(f"   API Key: {'✅ Set' if self.api_key else '❌ Not set'}")
        print(f"   Max Workers: {self.max_workers}")
        print(f"   Max Retries: {self.max_retries}")
        print(f"   External APIs: Jina {'✅' if self.jina_api_key else '❌'}")
    
    def update_from_args(self, args):
        """Update configuration with command line arguments"""
        if hasattr(args, 'judge_model') and args.judge_model:
            self.judge_model = args.judge_model
        if hasattr(args, 'max_retries') and args.max_retries is not None:
            self.max_retries = args.max_retries
        if hasattr(args, 'max_workers') and args.max_workers is not None:
            self.max_workers = args.max_workers
        if hasattr(args, 'openai_api_key') and args.openai_api_key:
            self.api_key = args.api_key
        if hasattr(args, 'jina_api_key') and args.jina_api_key:
            self.jina_api_key = args.jina_api_key
        if hasattr(args, 'max_retries') and args.max_retries is not None:
            self.max_retries = args.max_retries
        if hasattr(args, 'max_workers') and args.max_workers is not None:
            self.max_workers = args.max_workers
        
        return self


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config():
    """Reload configuration"""
    global _config
    _config = Config()
