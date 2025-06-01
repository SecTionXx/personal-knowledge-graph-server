"""Configuration management for Personal Knowledge Graph Server."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

@dataclass
class OpenRouterConfig:
    """OpenRouter API configuration."""
    api_key: str
    base_url: str
    models: Dict[str, str]
    max_tokens: int
    temperature: float

@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str
    username: str
    password: str
    database: str
    max_connections: int
    timeout: int

@dataclass
class FileMonitoringConfig:
    """File monitoring configuration."""
    watch_directory: str
    processed_directory: str
    inbox_directory: str
    supported_extensions: List[str]
    processing_delay: int
    max_file_size: int

@dataclass
class BudgetConfig:
    """Budget and cost management configuration."""
    daily_limit: float
    weekly_limit: float
    monthly_limit: float
    currency: str
    alert_threshold: float

@dataclass
class MCPServerConfig:
    """MCP server configuration."""
    port: int
    host: str
    max_concurrent_requests: int
    timeout: int

@dataclass
class EntityExtractionConfig:
    """Entity extraction configuration."""
    confidence_threshold: float
    max_entities_per_document: int
    entity_types: List[str]

@dataclass
class PrivacyConfig:
    """Privacy and security configuration."""
    sensitive_patterns: List[str]
    redaction_placeholder: str
    local_processing_directories: List[str]

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    file: str
    max_file_size: int
    backup_count: int
    format: str

class Config:
    """Main configuration class for the Personal Knowledge Graph Server."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config_data = self._load_config()
        
        # Initialize configuration sections
        self.openrouter = self._load_openrouter_config()
        self.neo4j = self._load_neo4j_config()
        self.file_monitoring = self._load_file_monitoring_config()
        self.budget = self._load_budget_config()
        self.mcp_server = self._load_mcp_server_config()
        self.entity_extraction = self._load_entity_extraction_config()
        self.privacy = self._load_privacy_config()
        self.logging = self._load_logging_config()
        
        # Validate configuration
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)
            return config_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var, data)
        else:
            return data
    
    def _load_openrouter_config(self) -> OpenRouterConfig:
        """Load OpenRouter configuration."""
        config = self._config_data.get("openrouter", {})
        return OpenRouterConfig(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", "https://openrouter.ai/api/v1"),
            models=config.get("models", {}),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.1)
        )
    
    def _load_neo4j_config(self) -> Neo4jConfig:
        """Load Neo4j configuration."""
        config = self._config_data.get("neo4j", {})
        return Neo4jConfig(
            uri=config.get("uri", "bolt://localhost:7687"),
            username=config.get("username", "neo4j"),
            password=config.get("password", ""),
            database=config.get("database", "personal-knowledge"),
            max_connections=config.get("max_connections", 10),
            timeout=config.get("timeout", 30)
        )
    
    def _load_file_monitoring_config(self) -> FileMonitoringConfig:
        """Load file monitoring configuration."""
        config = self._config_data.get("file_monitoring", {})
        return FileMonitoringConfig(
            watch_directory=config.get("watch_directory", "E:\\GraphKnowledge"),
            processed_directory=config.get("processed_directory", "E:\\GraphKnowledge\\processed"),
            inbox_directory=config.get("inbox_directory", "E:\\GraphKnowledge\\inbox"),
            supported_extensions=config.get("supported_extensions", [".md", ".txt", ".pdf"]),
            processing_delay=config.get("processing_delay", 2),
            max_file_size=config.get("max_file_size", 10485760)
        )
    
    def _load_budget_config(self) -> BudgetConfig:
        """Load budget configuration."""
        config = self._config_data.get("budget", {})
        return BudgetConfig(
            daily_limit=config.get("daily_limit", 3.00),
            weekly_limit=config.get("weekly_limit", 15.00),
            monthly_limit=config.get("monthly_limit", 50.00),
            currency=config.get("currency", "USD"),
            alert_threshold=config.get("alert_threshold", 0.8)
        )
    
    def _load_mcp_server_config(self) -> MCPServerConfig:
        """Load MCP server configuration."""
        config = self._config_data.get("mcp_server", {})
        return MCPServerConfig(
            port=config.get("port", 8080),
            host=config.get("host", "localhost"),
            max_concurrent_requests=config.get("max_concurrent_requests", 10),
            timeout=config.get("timeout", 300)
        )
    
    def _load_entity_extraction_config(self) -> EntityExtractionConfig:
        """Load entity extraction configuration."""
        config = self._config_data.get("entity_extraction", {})
        return EntityExtractionConfig(
            confidence_threshold=config.get("confidence_threshold", 0.7),
            max_entities_per_document=config.get("max_entities_per_document", 50),
            entity_types=config.get("entity_types", ["person", "organization", "concept"])
        )
    
    def _load_privacy_config(self) -> PrivacyConfig:
        """Load privacy configuration."""
        config = self._config_data.get("privacy", {})
        return PrivacyConfig(
            sensitive_patterns=config.get("sensitive_patterns", []),
            redaction_placeholder=config.get("redaction_placeholder", "[REDACTED]"),
            local_processing_directories=config.get("local_processing_directories", ["sensitive", "private"])
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration."""
        config = self._config_data.get("logging", {})
        return LoggingConfig(
            level=config.get("level", "INFO"),
            file=config.get("file", "logs/knowledge_graph.log"),
            max_file_size=config.get("max_file_size", 10485760),
            backup_count=config.get("backup_count", 5),
            format=config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    
    def _validate_config(self):
        """Validate the loaded configuration."""
        errors = []

        # Check if we're in demo mode (don't require external services)
        demo_mode = os.getenv('DEMO_MODE', 'false').lower() == 'true'
        
        # Validate OpenRouter API key (skip in demo mode)
        if not demo_mode and (not self.openrouter.api_key or self.openrouter.api_key.startswith("${")):
            errors.append("OpenRouter API key is not set. Please set OPENROUTER_API_KEY environment variable.")

        # Validate Neo4j password (skip in demo mode)
        if not demo_mode and (not self.neo4j.password or self.neo4j.password.startswith("${")):
            errors.append("Neo4j password is not set. Please set NEO4J_PASSWORD environment variable.")

        # Validate directories exist
        directories_to_check = [
            self.file_monitoring.watch_directory,
            self.file_monitoring.processed_directory,
            self.file_monitoring.inbox_directory
        ]

        for directory in directories_to_check:
            if directory and not directory.startswith("${") and not Path(directory).exists():
                try:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {directory}: {e}")

        # Validate budget limits
        if self.budget.daily_limit <= 0:
            errors.append("Daily budget limit must be positive.")

        if errors and not demo_mode:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        elif errors and demo_mode:
            logger.warning("Running in demo mode with incomplete configuration:")
            for error in errors:
                logger.warning(f"  - {error}")
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        # Create logs directory if it doesn't exist
        log_file_path = Path(self.logging.file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            handlers=[
                logging.FileHandler(self.logging.file),
                logging.StreamHandler()
            ]
        )
    
    def get_model_for_task(self, task_complexity: str = "simple") -> str:
        """Get the appropriate model for a given task complexity.
        
        Args:
            task_complexity: "simple", "complex", or "batch"
            
        Returns:
            Model name for the task
        """
        return self.openrouter.models.get(task_complexity, self.openrouter.models.get("simple"))
    
    def is_sensitive_directory(self, file_path: str) -> bool:
        """Check if a file path is in a sensitive directory.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if the file is in a sensitive directory
        """
        path = Path(file_path)
        for sensitive_dir in self.privacy.local_processing_directories:
            if sensitive_dir in path.parts:
                return True
        return False

# Global configuration instance
config = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = Config()
    return config

def initialize_config(config_path: Optional[str] = None) -> Config:
    """Initialize the global configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Initialized configuration instance
    """
    global config
    config = Config(config_path)
    config.setup_logging()
    return config 