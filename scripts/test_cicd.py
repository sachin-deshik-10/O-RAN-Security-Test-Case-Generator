#!/usr/bin/env python3
"""
CI/CD Pipeline Testing Script
Author: N. Sachin Deshik
GitHub: sachin-deshik-10
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
import yaml


@dataclass
class TestResult:
    """Test result data class"""
    name: str
    status: str
    duration: float
    output: str
    error: Optional[str] = None


class CICDTester:
    """CI/CD Pipeline Testing and Validation"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.test_results: List[TestResult] = []
        self.start_time = time.time()
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str, str]:
        """Run a command and return success, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def test_python_environment(self) -> TestResult:
        """Test Python environment setup"""
        print("🐍 Testing Python environment...")
        start_time = time.time()
        
        try:
            # Check Python version
            success, stdout, stderr = self.run_command(["python", "--version"])
            if not success:
                return TestResult(
                    name="Python Environment",
                    status="FAILED",
                    duration=time.time() - start_time,
                    output=stdout,
                    error=stderr
                )
            
            # Check pip
            success, stdout, stderr = self.run_command(["pip", "--version"])
            if not success:
                return TestResult(
                    name="Python Environment",
                    status="FAILED",
                    duration=time.time() - start_time,
                    output=stdout,
                    error="pip not available"
                )
            
            # Check virtual environment
            venv_path = self.project_root / "venv"
            if not venv_path.exists():
                return TestResult(
                    name="Python Environment",
                    status="WARNING",
                    duration=time.time() - start_time,
                    output="Virtual environment not found",
                    error="No virtual environment detected"
                )
            
            return TestResult(
                name="Python Environment",
                status="PASSED",
                duration=time.time() - start_time,
                output=f"Python environment OK: {stdout.strip()}"
            )
            
        except Exception as e:
            return TestResult(
                name="Python Environment",
                status="FAILED",
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def test_dependencies(self) -> TestResult:
        """Test dependency installation"""
        print("📦 Testing dependencies...")
        start_time = time.time()
        
        try:
            # Check if requirements.txt exists
            requirements_file = self.project_root / "requirements.txt"
            if not requirements_file.exists():
                return TestResult(
                    name="Dependencies",
                    status="FAILED",
                    duration=time.time() - start_time,
                    output="",
                    error="requirements.txt not found"
                )
            
            # Try to install dependencies
            success, stdout, stderr = self.run_command([
                "pip", "install", "-r", "requirements.txt", "--dry-run"
            ])
            
            if not success:
                return TestResult(
                    name="Dependencies",
                    status="FAILED",
                    duration=time.time() - start_time,
                    output=stdout,
                    error=stderr
                )
            
            return TestResult(
                name="Dependencies",
                status="PASSED",
                duration=time.time() - start_time,
                output="All dependencies can be installed"
            )
            
        except Exception as e:
            return TestResult(
                name="Dependencies",
                status="FAILED",
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def test_code_quality(self) -> TestResult:
        """Test code quality with linting tools"""
        print("🔍 Testing code quality...")
        start_time = time.time()
        
        try:
            issues = []
            
            # Test with flake8
            success, stdout, stderr = self.run_command([
                "flake8", "ml_models/", "api/", "--max-line-length=88", "--extend-ignore=E203,W503"
            ])
            if not success and stderr:
                issues.append(f"flake8: {stderr}")
            
            # Test with black (check only)
            success, stdout, stderr = self.run_command([
                "black", "--check", "ml_models/", "api/"
            ])
            if not success and stderr:
                issues.append(f"black: {stderr}")
            
            # Test with isort (check only)
            success, stdout, stderr = self.run_command([
                "isort", "--check-only", "ml_models/", "api/"
            ])
            if not success and stderr:
                issues.append(f"isort: {stderr}")
            
            if issues:
                return TestResult(
                    name="Code Quality",
                    status="FAILED",
                    duration=time.time() - start_time,
                    output="Code quality issues found",
                    error="; ".join(issues)
                )
            
            return TestResult(
                name="Code Quality",
                status="PASSED",
                duration=time.time() - start_time,
                output="All code quality checks passed"
            )
            
        except Exception as e:
            return TestResult(
                name="Code Quality",
                status="FAILED",
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def test_security_scan(self) -> TestResult:
        """Test security scanning"""
        print("🔒 Testing security scan...")
        start_time = time.time()
        
        try:
            # Run bandit security scan
            success, stdout, stderr = self.run_command([
                "bandit", "-r", "ml_models/", "api/", "-f", "json"
            ])
            
            if not success:
                # Parse bandit output
                try:
                    results = json.loads(stdout)
                    high_severity = [r for r in results.get('results', []) if r['issue_severity'] == 'HIGH']
                    if high_severity:
                        return TestResult(
                            name="Security Scan",
                            status="FAILED",
                            duration=time.time() - start_time,
                            output=f"Found {len(high_severity)} high severity issues",
                            error=stderr
                        )
                except json.JSONDecodeError:
                    pass
            
            return TestResult(
                name="Security Scan",
                status="PASSED",
                duration=time.time() - start_time,
                output="No high severity security issues found"
            )
            
        except Exception as e:
            return TestResult(
                name="Security Scan",
                status="FAILED",
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def test_unit_tests(self) -> TestResult:
        """Test unit tests"""
        print("🧪 Testing unit tests...")
        start_time = time.time()
        
        try:
            # Run pytest
            success, stdout, stderr = self.run_command([
                "pytest", "tests/", "-v", "--tb=short", "--maxfail=5"
            ])
            
            if not success:
                return TestResult(
                    name="Unit Tests",
                    status="FAILED",
                    duration=time.time() - start_time,
                    output=stdout,
                    error=stderr
                )
            
            return TestResult(
                name="Unit Tests",
                status="PASSED",
                duration=time.time() - start_time,
                output="All unit tests passed"
            )
            
        except Exception as e:
            return TestResult(
                name="Unit Tests",
                status="FAILED",
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def test_api_endpoints(self) -> TestResult:
        """Test API endpoints"""
        print("🔌 Testing API endpoints...")
        start_time = time.time()
        
        try:
            # Start API server in background
            api_process = subprocess.Popen([
                "uvicorn", "api.oran_api:app", "--host", "0.0.0.0", "--port", "8000"
            ], cwd=self.project_root)
            
            # Wait for server to start
            time.sleep(10)
            
            # Test health endpoint
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code != 200:
                    return TestResult(
                        name="API Endpoints",
                        status="FAILED",
                        duration=time.time() - start_time,
                        output="",
                        error=f"Health check failed: {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                return TestResult(
                    name="API Endpoints",
                    status="FAILED",
                    duration=time.time() - start_time,
                    output="",
                    error=f"API server not responding: {e}"
                )
            finally:
                # Clean up
                api_process.terminate()
                api_process.wait()
            
            return TestResult(
                name="API Endpoints",
                status="PASSED",
                duration=time.time() - start_time,
                output="API endpoints responding correctly"
            )
            
        except Exception as e:
            return TestResult(
                name="API Endpoints",
                status="FAILED",
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def test_ml_models(self) -> TestResult:
        """Test ML models"""
        print("🤖 Testing ML models...")
        start_time = time.time()
        
        try:
            # Run ML model tests
            success, stdout, stderr = self.run_command([
                "pytest", "tests/test_ml_models.py", "-v"
            ])
            
            if not success:
                return TestResult(
                    name="ML Models",
                    status="FAILED",
                    duration=time.time() - start_time,
                    output=stdout,
                    error=stderr
                )
            
            return TestResult(
                name="ML Models",
                status="PASSED",
                duration=time.time() - start_time,
                output="All ML model tests passed"
            )
            
        except Exception as e:
            return TestResult(
                name="ML Models",
                status="FAILED",
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def test_docker_build(self) -> TestResult:
        """Test Docker build"""
        print("🐳 Testing Docker build...")
        start_time = time.time()
        
        try:
            # Check if Docker is available
            success, stdout, stderr = self.run_command(["docker", "--version"])
            if not success:
                return TestResult(
                    name="Docker Build",
                    status="SKIPPED",
                    duration=time.time() - start_time,
                    output="Docker not available",
                    error="Docker not installed"
                )
            
            # Build Docker image
            success, stdout, stderr = self.run_command([
                "docker", "build", "-t", "oran-security-test", "."
            ])
            
            if not success:
                return TestResult(
                    name="Docker Build",
                    status="FAILED",
                    duration=time.time() - start_time,
                    output=stdout,
                    error=stderr
                )
            
            return TestResult(
                name="Docker Build",
                status="PASSED",
                duration=time.time() - start_time,
                output="Docker image built successfully"
            )
            
        except Exception as e:
            return TestResult(
                name="Docker Build",
                status="FAILED",
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def test_documentation(self) -> TestResult:
        """Test documentation"""
        print("📖 Testing documentation...")
        start_time = time.time()
        
        try:
            required_files = [
                "README.md",
                "docs/architecture.md",
                "docs/api/",
                "CHANGELOG.md",
                "CONTRIBUTING.md"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not (self.project_root / file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                return TestResult(
                    name="Documentation",
                    status="FAILED",
                    duration=time.time() - start_time,
                    output="",
                    error=f"Missing documentation files: {', '.join(missing_files)}"
                )
            
            return TestResult(
                name="Documentation",
                status="PASSED",
                duration=time.time() - start_time,
                output="All required documentation files present"
            )
            
        except Exception as e:
            return TestResult(
                name="Documentation",
                status="FAILED",
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all tests"""
        print("🚀 Running Comprehensive CI/CD Tests")
        print("=" * 60)
        
        test_functions = [
            self.test_python_environment,
            self.test_dependencies,
            self.test_code_quality,
            self.test_security_scan,
            self.test_unit_tests,
            self.test_ml_models,
            self.test_api_endpoints,
            self.test_docker_build,
            self.test_documentation
        ]
        
        results = []
        for test_func in test_functions:
            result = test_func()
            results.append(result)
            self.test_results.append(result)
            
            # Print result
            status_emoji = {
                "PASSED": "✅",
                "FAILED": "❌",
                "WARNING": "⚠️",
                "SKIPPED": "⏭️"
            }
            print(f"{status_emoji.get(result.status, '❓')} {result.name}: {result.status}")
            if result.error:
                print(f"   Error: {result.error}")
            print(f"   Duration: {result.duration:.2f}s")
            print()
        
        return results
    
    def generate_report(self) -> Dict:
        """Generate test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASSED"])
        failed_tests = len([r for r in self.test_results if r.status == "FAILED"])
        skipped_tests = len([r for r in self.test_results if r.status == "SKIPPED"])
        warning_tests = len([r for r in self.test_results if r.status == "WARNING"])
        
        total_duration = time.time() - self.start_time
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "skipped": skipped_tests,
                "warnings": warning_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": total_duration
            },
            "tests": [
                {
                    "name": result.name,
                    "status": result.status,
                    "duration": result.duration,
                    "output": result.output,
                    "error": result.error
                }
                for result in self.test_results
            ]
        }
        
        return report
    
    def print_summary(self):
        """Print test summary"""
        report = self.generate_report()
        summary = report["summary"]
        
        print("\n" + "=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        print(f"⏭️  Skipped: {summary['skipped']}")
        print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Total Duration: {summary['total_duration']:.2f}s")
        
        if summary['failed'] > 0:
            print("\n❌ FAILED TESTS:")
            for test in report["tests"]:
                if test["status"] == "FAILED":
                    print(f"  • {test['name']}: {test['error']}")
        
        if summary['warnings'] > 0:
            print("\n⚠️  WARNINGS:")
            for test in report["tests"]:
                if test["status"] == "WARNING":
                    print(f"  • {test['name']}: {test['error']}")
        
        print("\n" + "=" * 60)
        
        # Save report
        report_file = self.project_root / "output" / "cicd_test_report.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_file}")


def main():
    """Main function"""
    print("🚀 O-RAN Security Test Case Generator")
    print("CI/CD Pipeline Testing Script")
    print("Author: N. Sachin Deshik")
    print("=" * 60)
    
    # Initialize tester
    tester = CICDTester()
    
    # Run tests
    results = tester.run_all_tests()
    
    # Print summary
    tester.print_summary()
    
    # Exit with appropriate code
    failed_count = len([r for r in results if r.status == "FAILED"])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
