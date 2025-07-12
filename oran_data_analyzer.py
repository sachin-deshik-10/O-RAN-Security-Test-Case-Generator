"""
O-RAN Data Analyzer and Quantitative Assessment Tool
Provides comprehensive analysis of O-RAN datasets with security and performance metrics
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ORANDataAnalyzer:
    """Comprehensive O-RAN Dataset Analysis Engine"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.datasets = {}
        self.analysis_results = {}
        self.load_datasets()
    
    def load_datasets(self):
        """Load all O-RAN datasets from data directory"""
        dataset_files = {
            'performance': 'near_rt_ric_performance.json',
            'security': 'xapp_security_metrics.json',
            'e2_traces': 'e2_interface_traces.json',
            'vulnerabilities': 'xapp_security_metrics.json',
            'oran_components': 'oran-components.json',
            'oran_near_rt_ric': 'oran-near-rt-ric.json',
            'oran_security_analysis': 'oran-security-analysis.json',
            'capec': 'capec.json',
            'cwe': 'cwe.json',
            'asvs': 'asvs.json'
        }
        
        for dataset_name, filename in dataset_files.items():
            try:
                with open(f"{self.data_dir}/{filename}", 'r', encoding='utf-8') as f:
                    self.datasets[dataset_name] = json.load(f)
                logger.info(f"Loaded dataset: {dataset_name}")
            except FileNotFoundError:
                logger.warning(f"Dataset file not found: {filename}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON in {filename}: {e}")
    
    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze Near-RT RIC performance metrics"""
        if 'performance' not in self.datasets:
            return {"error": "Performance dataset not available"}
        
        performance_data = self.datasets['performance']
        if 'metrics' not in performance_data:
            return {"error": "No metrics found in performance dataset"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(performance_data['metrics'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate statistical metrics
        analysis = {
            'dataset_info': {
                'total_records': len(df),
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'sampling_frequency': performance_data['metadata']['sampling_frequency']
            },
            'latency_analysis': {
                'e2e_latency': {
                    'mean': df['latency_e2e'].mean(),
                    'std': df['latency_e2e'].std(),
                    'min': df['latency_e2e'].min(),
                    'max': df['latency_e2e'].max(),
                    'p95': df['latency_e2e'].quantile(0.95),
                    'p99': df['latency_e2e'].quantile(0.99)
                },
                'e2_latency': {
                    'mean': df['latency_e2'].mean(),
                    'std': df['latency_e2'].std(),
                    'min': df['latency_e2'].min(),
                    'max': df['latency_e2'].max(),
                    'p95': df['latency_e2'].quantile(0.95),
                    'p99': df['latency_e2'].quantile(0.99)
                },
                'a1_latency': {
                    'mean': df['latency_a1'].mean(),
                    'std': df['latency_a1'].std(),
                    'min': df['latency_a1'].min(),
                    'max': df['latency_a1'].max(),
                    'p95': df['latency_a1'].quantile(0.95),
                    'p99': df['latency_a1'].quantile(0.99)
                },
                'o1_latency': {
                    'mean': df['latency_o1'].mean(),
                    'std': df['latency_o1'].std(),
                    'min': df['latency_o1'].min(),
                    'max': df['latency_o1'].max(),
                    'p95': df['latency_o1'].quantile(0.95),
                    'p99': df['latency_o1'].quantile(0.99)
                }
            },
            'throughput_analysis': {
                'data_throughput': {
                    'mean': df['throughput_data'].mean(),
                    'std': df['throughput_data'].std(),
                    'min': df['throughput_data'].min(),
                    'max': df['throughput_data'].max()
                },
                'message_throughput': {
                    'mean': df['throughput_messages'].mean(),
                    'std': df['throughput_messages'].std(),
                    'min': df['throughput_messages'].min(),
                    'max': df['throughput_messages'].max()
                }
            },
            'reliability_analysis': {
                'availability': {
                    'mean': df['availability'].mean(),
                    'std': df['availability'].std(),
                    'min': df['availability'].min(),
                    'max': df['availability'].max()
                },
                'error_rate': {
                    'mean': df['error_rate'].mean(),
                    'std': df['error_rate'].std(),
                    'min': df['error_rate'].min(),
                    'max': df['error_rate'].max()
                }
            },
            'resource_utilization': {
                'cpu_utilization': {
                    'mean': df['cpu_utilization'].mean(),
                    'std': df['cpu_utilization'].std(),
                    'min': df['cpu_utilization'].min(),
                    'max': df['cpu_utilization'].max()
                },
                'memory_utilization': {
                    'mean': df['memory_utilization'].mean(),
                    'std': df['memory_utilization'].std(),
                    'min': df['memory_utilization'].min(),
                    'max': df['memory_utilization'].max()
                }
            }
        }
        
        # Performance benchmarks (O-RAN Alliance specifications)
        benchmarks = {
            'e2e_latency_target': 10.0,  # ms
            'e2_latency_target': 5.0,    # ms
            'a1_latency_target': 100.0,  # ms
            'o1_latency_target': 1000.0, # ms
            'availability_target': 99.9,  # %
            'error_rate_target': 0.001   # %
        }
        
        # Compliance assessment
        compliance = {
            'e2e_latency_compliance': (df['latency_e2e'] <= benchmarks['e2e_latency_target']).mean(),
            'e2_latency_compliance': (df['latency_e2'] <= benchmarks['e2_latency_target']).mean(),
            'a1_latency_compliance': (df['latency_a1'] <= benchmarks['a1_latency_target']).mean(),
            'o1_latency_compliance': (df['latency_o1'] <= benchmarks['o1_latency_target']).mean(),
            'availability_compliance': (df['availability'] >= benchmarks['availability_target']).mean(),
            'error_rate_compliance': (df['error_rate'] <= benchmarks['error_rate_target']).mean()
        }
        
        analysis['benchmarks'] = benchmarks
        analysis['compliance'] = compliance
        analysis['overall_compliance'] = np.mean(list(compliance.values()))
        
        return analysis
    
    def get_basic_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """Get basic statistics for a dataset"""
        if dataset_name not in self.datasets:
            return {"error": f"Dataset '{dataset_name}' not found"}
        
        dataset = self.datasets[dataset_name]
        
        if isinstance(dataset, list) and len(dataset) > 0:
            # Convert to DataFrame for analysis
            try:
                df = pd.DataFrame(dataset)
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_columns) > 0:
                    stats = df[numeric_columns].describe().to_dict()
                    return {
                        'total_records': len(df),
                        'numeric_columns': len(numeric_columns),
                        'statistics': stats
                    }
                else:
                    return {
                        'total_records': len(df),
                        'numeric_columns': 0,
                        'statistics': {}
                    }
            except Exception as e:
                return {"error": f"Error processing dataset: {e}"}
        elif isinstance(dataset, dict):
            return {
                'total_keys': len(dataset),
                'keys': list(dataset.keys()),
                'type': 'dictionary'
            }
        else:
            return {
                'type': str(type(dataset)),
                'info': 'Non-tabular data'
            }
    
    def analyze_security_metrics(self, dataset_name: str = None) -> Dict[str, Any]:
        """Analyze security metrics and vulnerabilities"""
        if dataset_name and dataset_name in self.datasets:
            security_data = self.datasets[dataset_name]
        elif 'security' in self.datasets:
            security_data = self.datasets['security']
        else:
            return {"error": "Security dataset not available"}
        
        # Analyze vulnerabilities
        vulnerabilities = security_data.get('vulnerabilities', [])
        threat_patterns = security_data.get('threat_patterns', [])
        
        vuln_analysis = {
            'total_vulnerabilities': len(vulnerabilities),
            'critical_issues': 0,
            'security_score': 0.0,
            'severity_distribution': {},
            'component_analysis': {},
            'interface_analysis': {},
            'cvss_analysis': {}
        }
        
        if vulnerabilities:
            vuln_df = pd.DataFrame(vulnerabilities)
            
            # Count critical issues
            if 'severity' in vuln_df.columns:
                vuln_analysis['critical_issues'] = len(vuln_df[vuln_df['severity'] == 'Critical'])
                severity_counts = vuln_df['severity'].value_counts()
                vuln_analysis['severity_distribution'] = severity_counts.to_dict()
            
            # Calculate security score (0-10, higher is better)
            if 'cvss_score' in vuln_df.columns:
                avg_cvss = vuln_df['cvss_score'].mean()
                vuln_analysis['security_score'] = max(0, 10 - avg_cvss)
                
                vuln_analysis['cvss_analysis'] = {
                    'mean': avg_cvss,
                    'std': vuln_df['cvss_score'].std(),
                    'min': vuln_df['cvss_score'].min(),
                    'max': vuln_df['cvss_score'].max(),
                    'critical_count': len(vuln_df[vuln_df['cvss_score'] >= 9.0]),
                    'high_count': len(vuln_df[(vuln_df['cvss_score'] >= 7.0) & (vuln_df['cvss_score'] < 9.0)]),
                    'medium_count': len(vuln_df[(vuln_df['cvss_score'] >= 4.0) & (vuln_df['cvss_score'] < 7.0)]),
                    'low_count': len(vuln_df[vuln_df['cvss_score'] < 4.0])
                }
            else:
                # Fallback security score based on vulnerability count
                vuln_analysis['security_score'] = max(0, 10 - (len(vulnerabilities) / 10))
        
        return vuln_analysis
    
    def analyze_performance_metrics(self, dataset_name: str = None) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if dataset_name and dataset_name in self.datasets:
            perf_data = self.datasets[dataset_name]
        elif 'performance' in self.datasets:
            perf_data = self.datasets['performance']
        else:
            return {"error": "Performance dataset not available"}
        
        # Extract performance metrics
        metrics = perf_data.get('metrics', [])
        
        perf_analysis = {
            'avg_latency': 0.0,
            'max_latency': 0.0,
            'min_latency': 0.0,
            'throughput': 0.0,
            'availability': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'network_usage': 0.0
        }
        
        if metrics:
            df = pd.DataFrame(metrics)
            
            # Calculate latency metrics
            if 'latency' in df.columns:
                perf_analysis['avg_latency'] = df['latency'].mean()
                perf_analysis['max_latency'] = df['latency'].max()
                perf_analysis['min_latency'] = df['latency'].min()
            
            # Calculate throughput
            if 'throughput' in df.columns:
                perf_analysis['throughput'] = df['throughput'].mean()
            
            # Calculate availability
            if 'availability' in df.columns:
                perf_analysis['availability'] = df['availability'].mean()
            elif 'uptime' in df.columns:
                perf_analysis['availability'] = df['uptime'].mean()
            
            # Calculate resource usage
            if 'cpu_usage' in df.columns:
                perf_analysis['cpu_usage'] = df['cpu_usage'].mean()
            if 'memory_usage' in df.columns:
                perf_analysis['memory_usage'] = df['memory_usage'].mean()
            if 'network_usage' in df.columns:
                perf_analysis['network_usage'] = df['network_usage'].mean()
        
        return perf_analysis
    
    def create_security_dashboard(self, dataset_name: str = None) -> Optional[go.Figure]:
        """Create a security dashboard visualization"""
        try:
            security_metrics = self.analyze_security_metrics(dataset_name)
            
            if 'error' in security_metrics:
                return None
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Security Score', 'Vulnerability Distribution', 'CVSS Analysis', 'Component Analysis'),
                specs=[[{'type': 'indicator'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            # Security Score Gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=security_metrics.get('security_score', 0),
                    title={'text': "Security Score"},
                    gauge={'axis': {'range': [0, 10]},
                           'bar': {'color': "darkgreen"},
                           'steps': [{'range': [0, 4], 'color': "red"},
                                    {'range': [4, 7], 'color': "yellow"},
                                    {'range': [7, 10], 'color': "green"}]},
                    domain={'x': [0, 0.48], 'y': [0.5, 1]}
                ),
                row=1, col=1
            )
            
            # Vulnerability Distribution
            severity_dist = security_metrics.get('severity_distribution', {})
            if severity_dist:
                fig.add_trace(
                    go.Bar(
                        x=list(severity_dist.keys()),
                        y=list(severity_dist.values()),
                        name="Vulnerabilities",
                        marker_color=['red', 'orange', 'yellow', 'green']
                    ),
                    row=1, col=2
                )
            
            # CVSS Analysis
            cvss_data = security_metrics.get('cvss_analysis', {})
            if cvss_data:
                cvss_categories = ['Critical', 'High', 'Medium', 'Low']
                cvss_counts = [
                    cvss_data.get('critical_count', 0),
                    cvss_data.get('high_count', 0),
                    cvss_data.get('medium_count', 0),
                    cvss_data.get('low_count', 0)
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=cvss_categories,
                        y=cvss_counts,
                        name="CVSS Distribution",
                        marker_color=['darkred', 'red', 'orange', 'green']
                    ),
                    row=2, col=1
                )
            
            # Component Analysis
            component_data = security_metrics.get('component_analysis', {})
            if component_data:
                fig.add_trace(
                    go.Bar(
                        x=list(component_data.keys()),
                        y=list(component_data.values()),
                        name="Components",
                        marker_color='blue'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title_text="O-RAN Security Analysis Dashboard",
                showlegend=False,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating security dashboard: {e}")
            return None
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis on all datasets"""
        results = {}
        
        for dataset_name in self.datasets.keys():
            try:
                analysis = {
                    'basic_stats': self.get_basic_statistics(dataset_name),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add security analysis for security-related datasets
                if any(keyword in dataset_name.lower() for keyword in ['security', 'vulnerability', 'threat']):
                    analysis['security_metrics'] = self.analyze_security_metrics(dataset_name)
                
                # Add performance analysis for performance-related datasets
                if any(keyword in dataset_name.lower() for keyword in ['performance', 'metric', 'trace']):
                    analysis['performance_metrics'] = self.analyze_performance_metrics(dataset_name)
                
                results[dataset_name] = analysis
                
            except Exception as e:
                results[dataset_name] = {'error': str(e)}
        
        return results
        
        return {
            'vulnerabilities': vuln_analysis,
            'threat_patterns': threat_analysis,
            'security_score': self.calculate_security_score(vuln_analysis, threat_analysis)
        }
    
    def calculate_security_score(self, vuln_analysis: Dict, threat_analysis: Dict) -> float:
        """Calculate overall security score based on vulnerabilities and threats"""
        base_score = 100.0
        
        # Deduct points for vulnerabilities
        if vuln_analysis['total_vulnerabilities'] > 0:
            critical_penalty = vuln_analysis['cvss_analysis'].get('critical_count', 0) * 20
            high_penalty = vuln_analysis['cvss_analysis'].get('high_count', 0) * 10
            medium_penalty = vuln_analysis['cvss_analysis'].get('medium_count', 0) * 5
            low_penalty = vuln_analysis['cvss_analysis'].get('low_count', 0) * 2
            
            total_penalty = critical_penalty + high_penalty + medium_penalty + low_penalty
            base_score -= min(total_penalty, 80)  # Cap at 80% deduction
        
        # Deduct points for high-confidence threats
        if threat_analysis['total_patterns'] > 0:
            high_confidence_threats = threat_analysis['confidence_analysis'].get('high_confidence_count', 0)
            threat_penalty = high_confidence_threats * 5
            base_score -= min(threat_penalty, 20)  # Cap at 20% deduction
        
        return max(base_score, 0)
    
    def generate_performance_visualization(self) -> go.Figure:
        """Generate interactive performance visualization"""
        if 'performance' not in self.datasets:
            return None
        
        performance_data = self.datasets['performance']
        df = pd.DataFrame(performance_data['metrics'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Latency Metrics', 'Throughput Metrics', 'Reliability Metrics', 'Resource Utilization'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Latency metrics
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['latency_e2e'], name='E2E Latency', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['latency_e2'], name='E2 Latency', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['latency_a1'], name='A1 Latency', line=dict(color='green')), row=1, col=1, secondary_y=True)
        
        # Throughput metrics
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['throughput_data'], name='Data Throughput', line=dict(color='purple')), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['throughput_messages'], name='Message Throughput', line=dict(color='orange')), row=1, col=2)
        
        # Reliability metrics
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['availability'], name='Availability', line=dict(color='darkgreen')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['error_rate']*100, name='Error Rate (%)', line=dict(color='red')), row=2, col=1, secondary_y=True)
        
        # Resource utilization
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cpu_utilization'], name='CPU Utilization', line=dict(color='brown')), row=2, col=2)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['memory_utilization'], name='Memory Utilization', line=dict(color='pink')), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text="O-RAN Performance Metrics Dashboard")
        return fig
    
    def generate_security_visualization(self) -> go.Figure:
        """Generate interactive security visualization"""
        if 'security' not in self.datasets:
            return None
        
        security_data = self.datasets['security']
        vulnerabilities = security_data.get('vulnerabilities', [])
        threat_patterns = security_data.get('threat_patterns', [])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Vulnerability Severity Distribution', 'Component Vulnerability Analysis', 
                          'Threat Pattern Types', 'Attack Vector Distribution'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        if vulnerabilities:
            vuln_df = pd.DataFrame(vulnerabilities)
            
            # Vulnerability severity pie chart
            severity_counts = vuln_df['severity'].value_counts()
            fig.add_trace(go.Pie(labels=severity_counts.index, values=severity_counts.values, 
                               name="Severity"), row=1, col=1)
            
            # Component vulnerability bar chart
            component_counts = vuln_df['component'].value_counts()
            fig.add_trace(go.Bar(x=component_counts.index, y=component_counts.values, 
                               name="Components"), row=1, col=2)
        
        if threat_patterns:
            pattern_df = pd.DataFrame(threat_patterns)
            
            # Threat pattern types pie chart
            pattern_type_counts = pattern_df['pattern_type'].value_counts()
            fig.add_trace(go.Pie(labels=pattern_type_counts.index, values=pattern_type_counts.values, 
                               name="Pattern Types"), row=2, col=1)
            
            # Attack vector bar chart
            vector_counts = pattern_df['attack_vector'].value_counts()
            fig.add_trace(go.Bar(x=vector_counts.index, y=vector_counts.values, 
                               name="Attack Vectors"), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text="O-RAN Security Analysis Dashboard")
        return fig
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        performance_analysis = self.analyze_performance_metrics()
        security_analysis = self.analyze_security_metrics()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'overall_score': 0,
                'performance_compliance': 0,
                'security_score': 0,
                'data_quality': 0
            },
            'performance': performance_analysis,
            'security': security_analysis,
            'recommendations': []
        }
        
        # Calculate overall scores
        if 'error' not in performance_analysis:
            report['summary']['performance_compliance'] = performance_analysis.get('overall_compliance', 0) * 100
        
        report['summary']['security_score'] = security_analysis.get('security_score', 0)
        report['summary']['data_quality'] = 90  # Placeholder - would be calculated based on data validation
        
        # Overall score calculation
        report['summary']['overall_score'] = (
            report['summary']['performance_compliance'] * 0.4 +
            report['summary']['security_score'] * 0.4 +
            report['summary']['data_quality'] * 0.2
        )
        
        # Generate recommendations
        recommendations = []
        
        if report['summary']['performance_compliance'] < 80:
            recommendations.append({
                'category': 'Performance',
                'priority': 'HIGH',
                'title': 'Improve Performance Compliance',
                'description': 'Performance metrics are below O-RAN compliance thresholds',
                'actions': ['Optimize latency bottlenecks', 'Increase resource allocation', 'Review system architecture']
            })
        
        if report['summary']['security_score'] < 70:
            recommendations.append({
                'category': 'Security',
                'priority': 'CRITICAL',
                'title': 'Address Security Vulnerabilities',
                'description': 'Critical security issues detected',
                'actions': ['Patch critical vulnerabilities', 'Implement additional security controls', 'Conduct security audit']
            })
        
        report['recommendations'] = recommendations
        
        return report
    
    def export_analysis_results(self, filename: str = None) -> str:
        """Export analysis results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"oran_analysis_report_{timestamp}.json"
        
        report = self.generate_compliance_report()
        
        # Ensure output directory exists
        import os
        os.makedirs("./output", exist_ok=True)
        
        filepath = f"./output/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filepath

def create_streamlit_dashboard():
    """Create Streamlit dashboard for O-RAN data analysis"""
    st.set_page_config(page_title="O-RAN Data Analysis Dashboard", layout="wide")
    
    st.title("🛡️ O-RAN Security & Performance Analysis Dashboard")
    st.markdown("Comprehensive analysis of O-RAN datasets with quantitative metrics and security assessment")
    
    # Initialize analyzer
    analyzer = ORANDataAnalyzer()
    
    # Sidebar controls
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Performance Analysis", "Security Analysis", "Compliance Report", "Full Dashboard"]
    )
    
    if analysis_type == "Performance Analysis":
        st.header("📊 Performance Metrics Analysis")
        
        performance_analysis = analyzer.analyze_performance_metrics()
        
        if 'error' in performance_analysis:
            st.error(performance_analysis['error'])
        else:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Compliance", f"{performance_analysis['overall_compliance']:.1%}")
            
            with col2:
                st.metric("Total Records", performance_analysis['dataset_info']['total_records'])
            
            with col3:
                e2e_mean = performance_analysis['latency_analysis']['e2e_latency']['mean']
                st.metric("Avg E2E Latency", f"{e2e_mean:.2f} ms")
            
            with col4:
                availability_mean = performance_analysis['reliability_analysis']['availability']['mean']
                st.metric("Avg Availability", f"{availability_mean:.2f}%")
            
            # Performance visualization
            fig = analyzer.generate_performance_visualization()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            st.subheader("Detailed Performance Metrics")
            st.json(performance_analysis)
    
    elif analysis_type == "Security Analysis":
        st.header("🔒 Security Assessment")
        
        security_analysis = analyzer.analyze_security_metrics()
        
        if 'error' in security_analysis:
            st.error(security_analysis['error'])
        else:
            # Display security metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Security Score", f"{security_analysis['security_score']:.1f}/100")
            
            with col2:
                st.metric("Total Vulnerabilities", security_analysis['vulnerabilities']['total_vulnerabilities'])
            
            with col3:
                st.metric("Threat Patterns", security_analysis['threat_patterns']['total_patterns'])
            
            with col4:
                critical_count = security_analysis['vulnerabilities']['cvss_analysis'].get('critical_count', 0)
                st.metric("Critical Issues", critical_count)
            
            # Security visualization
            fig = analyzer.generate_security_visualization()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed security analysis
            st.subheader("Detailed Security Analysis")
            st.json(security_analysis)
    
    elif analysis_type == "Compliance Report":
        st.header("📋 Compliance Report")
        
        report = analyzer.generate_compliance_report()
        
        # Summary metrics
        st.subheader("Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Score", f"{report['summary']['overall_score']:.1f}/100")
        
        with col2:
            st.metric("Performance Compliance", f"{report['summary']['performance_compliance']:.1f}%")
        
        with col3:
            st.metric("Security Score", f"{report['summary']['security_score']:.1f}/100")
        
        with col4:
            st.metric("Data Quality", f"{report['summary']['data_quality']:.1f}/100")
        
        # Recommendations
        st.subheader("Recommendations")
        for rec in report['recommendations']:
            with st.expander(f"[{rec['priority']}] {rec['title']}"):
                st.write(rec['description'])
                st.write("**Actions:**")
                for action in rec['actions']:
                    st.write(f"- {action}")
        
        # Export functionality
        if st.button("Export Report"):
            filepath = analyzer.export_analysis_results()
            st.success(f"Report exported to: {filepath}")
    
    elif analysis_type == "Full Dashboard":
        st.header("🎯 Complete O-RAN Analysis Dashboard")
        
        # Performance section
        st.subheader("Performance Metrics")
        performance_analysis = analyzer.analyze_performance_metrics()
        
        if 'error' not in performance_analysis:
            fig_perf = analyzer.generate_performance_visualization()
            if fig_perf:
                st.plotly_chart(fig_perf, use_container_width=True)
        
        # Security section
        st.subheader("Security Assessment")
        security_analysis = analyzer.analyze_security_metrics()
        
        if 'error' not in security_analysis:
            fig_sec = analyzer.generate_security_visualization()
            if fig_sec:
                st.plotly_chart(fig_sec, use_container_width=True)
        
        # Compliance summary
        st.subheader("Compliance Summary")
        report = analyzer.generate_compliance_report()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", f"{report['summary']['overall_score']:.1f}/100")
        with col2:
            st.metric("Performance", f"{report['summary']['performance_compliance']:.1f}%")
        with col3:
            st.metric("Security", f"{report['summary']['security_score']:.1f}/100")

if __name__ == "__main__":
    # Run the Streamlit dashboard
    create_streamlit_dashboard()
