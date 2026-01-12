#!/usr/bin/env python3
"""
L4D2-AI-Architect System Validation Script

Tests all major components to ensure the system is ready for deployment.
Run this locally or on Vultr after setup to verify everything works.
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Colors for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")

def test_passed(msg):
    print(f"{GREEN}✓{RESET} {msg}")
    return True

def test_failed(msg):
    print(f"{RED}✗{RESET} {msg}")
    return False

def test_warning(msg):
    print(f"{YELLOW}⚠{RESET} {msg}")
    return True

class SystemValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.resolve()  # Go up to project root
        # Security: Validate project root exists and is a directory
        if not self.project_root.is_dir():
            raise RuntimeError(f"Invalid project root: {self.project_root}")
        self.tests_passed = 0
        self.tests_failed = 0
        self.warnings = 0
        
    def check_python_version(self):
        """Check Python version is 3.10+"""
        print_header("Python Version Check")
        version = sys.version_info
        if version.major == 3 and version.minor >= 10:
            return test_passed(f"Python {version.major}.{version.minor}.{version.micro} detected")
        else:
            return test_failed(f"Python 3.10+ required, found {version.major}.{version.minor}")
    
    def check_gpu(self):
        """Check GPU availability"""
        print_header("GPU Check")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                test_passed(f"GPU detected: {gpu_name}")
                test_passed(f"GPU memory: {gpu_memory:.1f} GB")
                return True
            else:
                test_warning("No GPU detected - CPU mode will be used (slower)")
                return True
        except ImportError:
            return test_failed("PyTorch not installed")
        except Exception as e:
            return test_failed(f"GPU check error: {e}")
    
    def check_dependencies(self):
        """Check critical Python dependencies"""
        print_header("Dependencies Check")
        dependencies = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("datasets", "Datasets"),
            ("peft", "PEFT"),
            ("gymnasium", "Gymnasium"),
            ("stable_baselines3", "Stable-Baselines3"),
            ("requests", "Requests"),
            ("beautifulsoup4", "BeautifulSoup"),
            ("fastapi", "FastAPI"),
            ("uvicorn", "Uvicorn"),
        ]
        
        all_good = True
        for package, name in dependencies:
            try:
                __import__(package)
                test_passed(f"{name} installed")
            except ImportError:
                test_failed(f"{name} not installed")
                all_good = False
        
        return all_good
    
    def check_directory_structure(self):
        """Check project directory structure"""
        print_header("Directory Structure Check")
        
        required_dirs = [
            "scripts/scrapers",
            "scripts/training", 
            "scripts/rl_training",
            "scripts/director",
            "scripts/inference",
            "scripts/utils",
            "configs",
            "data",
            "docs",
        ]
        
        all_good = True
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                test_passed(f"Directory exists: {dir_path}")
            else:
                test_failed(f"Missing directory: {dir_path}")
                all_good = False
        
        return all_good
    
    def check_config_files(self):
        """Check configuration files"""
        print_header("Configuration Files Check")
        
        config_files = [
            "configs/unsloth_config.yaml",
            "configs/director_config.yaml",
            ".env.example",
            "requirements.txt",
        ]
        
        all_good = True
        for config_file in config_files:
            full_path = self.project_root / config_file
            if full_path.exists():
                test_passed(f"Config exists: {config_file}")
            else:
                test_failed(f"Missing config: {config_file}")
                all_good = False
        
        # Check for .env
        env_path = self.project_root / ".env"
        if env_path.exists():
            test_passed(".env file configured")
        else:
            test_warning(".env not found - copy from .env.example and configure")
        
        return all_good
    
    def check_scripts(self):
        """Check critical scripts exist and are executable"""
        print_header("Scripts Check")
        
        scripts = [
            "setup.sh",
            "activate.sh",
            "run_scraping.sh",
            "run_training.sh",
            "scripts/scrapers/scrape_github_plugins.py",
            "scripts/scrapers/scrape_valve_wiki.py",
            "scripts/training/prepare_dataset.py",
            "scripts/training/train_unsloth.py",
            "scripts/training/export_model.py",
            "scripts/rl_training/train_ppo.py",
            "scripts/director/director.py",
            "scripts/director/bridge.py",
            "scripts/director/policy.py",
            "scripts/inference/copilot_server.py",
            "scripts/inference/copilot_cli.py",
            "scripts/utils/security.py",
            "scripts/utils/vultr_setup.sh",
        ]
        
        all_good = True
        for script in scripts:
            full_path = self.project_root / script
            if full_path.exists():
                test_passed(f"Script exists: {script}")
            else:
                test_failed(f"Missing script: {script}")
                all_good = False
        
        return all_good
    
    def test_imports(self):
        """Test critical imports"""
        print_header("Import Tests")
        
        test_imports = [
            ("scripts.scrapers.scrape_github_plugins", "GitHub scraper"),
            ("scripts.scrapers.scrape_valve_wiki", "Valve Wiki scraper"),
            ("scripts.training.prepare_dataset", "Dataset preparation"),
            ("scripts.director.director", "AI Director"),
            ("scripts.director.bridge", "Game bridge"),
            ("scripts.inference.copilot_server", "Copilot server"),
            ("scripts.utils.security", "Security utilities"),
        ]
        
        # Add scripts to path (validated project_root only)
        project_str = str(self.project_root)
        if project_str not in sys.path:
            sys.path.insert(0, project_str)
        
        all_good = True
        for module, name in test_imports:
            try:
                __import__(module)
                test_passed(f"{name} imports successfully")
            except ImportError as e:
                test_failed(f"{name} import failed: {e}")
                all_good = False
            except Exception as e:
                test_warning(f"{name} import warning: {e}")
        
        return all_good
    
    def test_network_services(self):
        """Test network services can be started"""
        print_header("Network Services Test")
        
        # Test if ports are available
        import socket
        
        ports_to_test = [
            (8000, "Copilot API"),
            (27050, "Director Bridge"),
            (6006, "TensorBoard"),
        ]
        
        all_good = True
        for port, service in ports_to_test:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result == 0:
                test_warning(f"Port {port} ({service}) is already in use")
            else:
                test_passed(f"Port {port} ({service}) is available")
        
        return all_good
    
    def test_data_pipeline(self):
        """Test data collection and processing pipeline"""
        print_header("Data Pipeline Test")
        
        # Check if we can import and use scrapers
        try:
            from scripts.scrapers.scrape_github_plugins import GitHubScraper
            from scripts.scrapers.scrape_valve_wiki import WikiScraper
            # Test instantiation
            github_scraper = GitHubScraper(token=None)
            wiki_scraper = WikiScraper()
            assert github_scraper is not None
            assert wiki_scraper is not None
            test_passed("Scrapers can be instantiated")
            
            # Check security functions
            from scripts.utils.security import validate_url, safe_path
            validated_url = validate_url("https://api.github.com", ["api.github.com"])
            validated_path = safe_path("test.txt", self.project_root)
            assert validated_url == "https://api.github.com"
            assert validated_path is not None
            test_passed("Security functions working")
            
            return True
        except Exception as e:
            return test_failed(f"Data pipeline test failed: {e}")
    
    def test_model_inference(self):
        """Test model inference components"""
        print_header("Model Inference Test")
        
        try:
            # Test copilot CLI
            from scripts.inference.copilot_cli import CopilotClient, validate_server_url
            
            # Test URL validation
            validate_server_url("http://localhost:8000")
            test_passed("Copilot URL validation working")
            
            # Test client creation
            client = CopilotClient("http://localhost:8000")
            assert client.base_url == "http://localhost:8000"
            test_passed("Copilot client can be created")
            
            return True
        except Exception as e:
            return test_failed(f"Inference test failed: {e}")
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("\n" + "="*60)
        print("  L4D2-AI-Architect System Validation")
        print("="*60)
        
        tests = [
            self.check_python_version(),
            self.check_gpu(),
            self.check_dependencies(),
            self.check_directory_structure(),
            self.check_config_files(),
            self.check_scripts(),
            self.test_imports(),
            self.test_network_services(),
            self.test_data_pipeline(),
            self.test_model_inference(),
        ]
        
        self.tests_passed = sum(1 for t in tests if t)
        self.tests_failed = len(tests) - self.tests_passed
        
        # Summary
        print("\n" + "="*60)
        print("  Validation Summary")
        print("="*60)
        
        if self.tests_failed == 0:
            print(f"\n{GREEN}✓ All tests passed!{RESET}")
            print("\nSystem is ready for deployment.")
            print("\nNext steps:")
            print("1. Configure .env file with API keys")
            print("2. Run data collection: ./run_scraping.sh")
            print("3. Start training: ./run_training.sh")
            print("4. Deploy services: systemctl start l4d2-copilot")
        else:
            print(f"\n{RED}✗ {self.tests_failed} test(s) failed{RESET}")
            print("\nPlease fix the issues above before proceeding.")
            print("\nCommon fixes:")
            print("- Install missing dependencies: pip install -r requirements.txt")
            print("- Run setup script: ./setup.sh")
            print("- Clone missing files from GitHub")
        
        return self.tests_failed == 0


def main():
    """Main validation entry point"""
    validator = SystemValidator()
    success = validator.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
