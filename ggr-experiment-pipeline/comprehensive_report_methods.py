    def generate_comprehensive_report(self, experiment_results: Dict[str, Any], output_folder: str) -> str:
        """Generate a comprehensive experiment report in Markdown format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_folder, "comprehensive_experiment_report.md")
        
        with open(report_file, 'w') as f:
            self._write_report_header(f, experiment_results, timestamp)
            self._write_table_of_contents(f)
            self._write_experiment_configuration(f, experiment_results)
            self._write_performance_summary(f, experiment_results)
            self._write_vllm_metrics_analysis(f, experiment_results)
            self._write_resource_utilization(f, experiment_results)
            self._write_batch_analysis(f, experiment_results)
            self._write_kv_cache_visualizations(f, experiment_results, output_folder)
            self._write_analysis_and_recommendations(f, experiment_results)
            self._write_data_exports(f, output_folder)
        
        logger.info(f"📊 Comprehensive report generated: {report_file}")
        return report_file
    
    def _write_report_header(self, f, experiment_results: Dict[str, Any], timestamp: str):
        """Write the report header section"""
        exp_info = experiment_results.get('experiment_info', {})
        f.write(f"# 📊 Comprehensive vLLM Experiment Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}  \n")
        f.write(f"**Run Folder**: `{os.path.basename(exp_info.get('dataset_path', 'unknown'))}`  \n")
        f.write(f"**Experiment**: {exp_info.get('query_key', 'N/A')} on {os.path.basename(exp_info.get('dataset_path', 'unknown'))}  \n\n")
        
        # Dataset type detection
        dataset_name = os.path.basename(exp_info.get('dataset_path', '')).lower()
        if any(keyword in dataset_name for keyword in ['reordered', 'ggr', 'ordered', 'sorted']):
            f.write(f"🚀 **Dataset Type**: Reordered/Optimized (likely using GGR or similar algorithm)  \n")
        elif any(keyword in dataset_name for keyword in ['shuffled', 'random', 'baseline']):
            f.write(f"🔀 **Dataset Type**: Shuffled/Random (baseline comparison)  \n")
        else:
            f.write(f"📁 **Dataset Type**: Original/Natural order  \n")
        f.write(f"\n")
    
    def _write_table_of_contents(self, f):
        """Write the table of contents"""
        f.write(f"## 📋 Table of Contents\n\n")
        f.write(f"1. [Experiment Configuration](#1-experiment-configuration)\n")
        f.write(f"2. [Performance Summary](#2-performance-summary)\n")
        f.write(f"3. [vLLM Engine Metrics](#3-vllm-engine-metrics)\n")
        f.write(f"4. [Resource Utilization](#4-resource-utilization)\n")
        f.write(f"5. [Batch Processing Analysis](#5-batch-processing-analysis)\n")
        f.write(f"6. [KV Cache Visualizations](#6-kv-cache-visualizations)\n")
        f.write(f"7. [Analysis & Recommendations](#7-analysis--recommendations)\n")
        f.write(f"8. [Data Exports](#8-data-exports)\n\n")
    
    def _write_experiment_configuration(self, f, experiment_results: Dict[str, Any]):
        """Write experiment configuration section"""
        exp_info = experiment_results.get('experiment_info', {})
        f.write(f"## 1. Experiment Configuration\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|-----------|-------|\n")
        f.write(f"| **Model** | {exp_info.get('model_name', 'N/A')} |\n")
        f.write(f"| **Dataset** | {os.path.basename(exp_info.get('dataset_path', 'N/A'))} |\n")
        f.write(f"| **Query Key** | {exp_info.get('query_key', 'N/A')} |\n")
        f.write(f"| **Query Type** | {exp_info.get('query_type', 'N/A')} |\n")
        f.write(f"| **GPU ID** | {exp_info.get('gpu_id', 'N/A')} |\n")
        f.write(f"| **Total Rows** | {exp_info.get('total_rows', 'N/A'):,} |\n")
        f.write(f"| **Processed Rows** | {exp_info.get('processed_rows', 'N/A'):,} |\n")
        f.write(f"| **Batch Size** | {exp_info.get('batch_size', 'N/A')} |\n")
        f.write(f"| **Experiment Time** | {exp_info.get('experiment_timestamp', 'N/A')} |\n\n")
    
    def _write_performance_summary(self, f, experiment_results: Dict[str, Any]):
        """Write performance summary section"""
        perf = experiment_results.get('performance_metrics', {})
        f.write(f"## 2. Performance Summary\n\n")
        f.write(f"### ⏱️ Timing Metrics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Total Inference Time** | {perf.get('total_inference_time', 0):.2f}s |\n")
        f.write(f"| **Prompt Creation Time** | {perf.get('prompt_creation_time', 0):.2f}s |\n")
        f.write(f"| **Average Time per Row** | {perf.get('avg_time_per_row', 0):.3f}s |\n")
        f.write(f"| **Overall Throughput** | {perf.get('overall_throughput_tokens_per_sec', 0):.1f} tokens/sec |\n\n")
        
        f.write(f"### 🔢 Token Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Estimated Total Tokens** | {perf.get('estimated_total_tokens', 0):,} |\n")
        f.write(f"| **Input Tokens** | {perf.get('estimated_input_tokens', 0):,} |\n")
        f.write(f"| **Output Tokens** | {perf.get('estimated_output_tokens', 0):,} |\n")
        f.write(f"| **Successful Batches** | {perf.get('successful_batches', 0)} |\n")
        f.write(f"| **Failed Batches** | {perf.get('failed_batches', 0)} |\n\n")
    
    def _write_vllm_metrics_analysis(self, f, experiment_results: Dict[str, Any]):
        """Write vLLM metrics analysis section"""
        vllm_metrics = experiment_results.get('vllm_metrics', {})
        f.write(f"## 3. vLLM Engine Metrics\n\n")
        f.write(f"### 🔧 Collection Status\n\n")
        f.write(f"- **Prefix Caching Enabled**: {vllm_metrics.get('prefix_caching_enabled', True)}\n")
        
        initial_stats = vllm_metrics.get('initial_stats', {})
        final_stats = vllm_metrics.get('final_stats', {})
        
        if final_stats.get('vllm_stats_available', False):
            key_metrics = final_stats.get('key_metrics', {})
            collection_methods = final_stats.get('collection_methods_tried', [])
            f.write(f"- **Collection Methods**: {', '.join(collection_methods)}\n")
            f.write(f"- **Metrics Available**: ✅ Yes\n\n")
            
            self._write_kv_cache_analysis(f, key_metrics, experiment_results)
            self._write_token_processing_stats(f, key_metrics)
            self._write_request_queue_stats(f, key_metrics)
            
        else:
            f.write(f"- **Metrics Available**: ❌ No\n")
            f.write(f"- **Error**: {final_stats.get('error', 'Unknown error')}\n")
            if 'collection_methods_tried' in final_stats:
                f.write(f"- **Methods Tried**: {', '.join(final_stats['collection_methods_tried'])}\n")
            f.write(f"\n")
    
    def _write_kv_cache_analysis(self, f, key_metrics: Dict[str, Any], experiment_results: Dict[str, Any]):
        """Write KV cache analysis subsection"""
        f.write(f"### 🎯 KV Cache Performance\n\n")
        
        # GPU Cache Usage
        if 'gpu_cache_usage_percent' in key_metrics:
            f.write(f"- **GPU Cache Usage**: {key_metrics['gpu_cache_usage_percent']:.1f}%\n")
        elif 'gpu_cache_usage_sys' in key_metrics:
            f.write(f"- **GPU Cache Usage (sys)**: {key_metrics['gpu_cache_usage_sys']:.1f}%\n")
        
        # Prefix Cache Hit Rate Analysis
        hit_rate_found = False
        if 'gpu_prefix_cache_hit_rate_percent' in key_metrics:
            hit_rate = key_metrics['gpu_prefix_cache_hit_rate_percent']
            hits = key_metrics.get('gpu_prefix_cache_hits', 0)
            queries = key_metrics.get('gpu_prefix_cache_queries', 0)
            
            # Validate hit rate data
            if queries > 10 or hit_rate < 99:  # Avoid reporting 100% from tiny values
                hit_rate_found = True
                if hits > 0 and queries > 0:
                    f.write(f"- **GPU Prefix Cache Hit Rate**: {hit_rate:.2f}% ({hits:,} hits / {queries:,} queries)\n")
                else:
                    f.write(f"- **GPU Prefix Cache Hit Rate**: {hit_rate:.2f}%\n")
                
                # Performance analysis
                dataset_name = os.path.basename(experiment_results.get('experiment_info', {}).get('dataset_path', '')).lower()
                is_reordered = any(keyword in dataset_name for keyword in ['reordered', 'ggr', 'ordered', 'sorted'])
                
                if hit_rate > 70:
                    f.write(f"  - 🏆 **EXCELLENT** - Very high hit rate indicates extremely effective data ordering!\n")
                    if is_reordered:
                        f.write(f"  - The optimized data ordering is providing substantial prefix reuse benefits.\n")
                elif hit_rate > 40:
                    f.write(f"  - ✅ **GOOD** - High hit rate shows effective prefix caching benefits!\n")
                elif hit_rate > 15:
                    f.write(f"  - ⚠️ **MODERATE** - Some prefix reuse detected, room for optimization.\n")
                elif hit_rate > 5:
                    f.write(f"  - ⚠️ **LOW** - Minimal prefix reuse, limited caching effectiveness.\n")
                else:
                    f.write(f"  - ❌ **VERY LOW** - Almost no prefix reuse detected.\n")
        
        if not hit_rate_found:
            f.write(f"- **Prefix Cache Hit Rate**: Not available\n")
            f.write(f"  - This may be due to:\n")
            f.write(f"    • vLLM version compatibility (try setting VLLM_USE_V1=0)\n")
            f.write(f"    • Insufficient requests processed yet\n")
            f.write(f"    • Prefix caching not fully enabled\n")
        
        f.write(f"\n")
    
    def _write_token_processing_stats(self, f, key_metrics: Dict[str, Any]):
        """Write token processing statistics"""
        f.write(f"### 🔤 Token Processing\n\n")
        if 'prompt_tokens_total' in key_metrics:
            f.write(f"- **Total Prompt Tokens**: {key_metrics['prompt_tokens_total']:,}\n")
        if 'generation_tokens_total' in key_metrics:
            f.write(f"- **Total Generation Tokens**: {key_metrics['generation_tokens_total']:,}\n")
        
        # Performance latencies if available
        histogram_summary = key_metrics.get('histogram_summary', {})
        if histogram_summary:
            f.write(f"\n#### Performance Latencies\n")
            if 'avg_time_to_first_token_seconds' in histogram_summary:
                f.write(f"- **Average Time to First Token**: {histogram_summary['avg_time_to_first_token_seconds']:.3f}s\n")
            if 'avg_time_per_output_token_seconds' in histogram_summary:
                f.write(f"- **Average Time per Output Token**: {histogram_summary['avg_time_per_output_token_seconds']:.4f}s\n")
            if 'avg_e2e_latency_seconds' in histogram_summary:
                f.write(f"- **Average E2E Request Latency**: {histogram_summary['avg_e2e_latency_seconds']:.3f}s\n")
        
        f.write(f"\n")
    
    def _write_request_queue_stats(self, f, key_metrics: Dict[str, Any]):
        """Write request queue statistics"""
        f.write(f"### 📊 Request Queue Status\n\n")
        if 'requests_running' in key_metrics:
            f.write(f"- **Running Requests**: {key_metrics['requests_running']}\n")
        elif 'num_requests_running' in key_metrics:
            f.write(f"- **Running Requests**: {key_metrics['num_requests_running']}\n")
        
        if 'requests_waiting' in key_metrics:
            f.write(f"- **Waiting Requests**: {key_metrics['requests_waiting']}\n")
        elif 'num_requests_waiting' in key_metrics:
            f.write(f"- **Waiting Requests**: {key_metrics['num_requests_waiting']}\n")
        
        f.write(f"\n")
    
    def _write_resource_utilization(self, f, experiment_results: Dict[str, Any]):
        """Write resource utilization section"""
        resource_data = experiment_results.get('resource_monitoring', {})
        if not resource_data:
            f.write(f"## 4. Resource Utilization\n\n")
            f.write(f"Resource monitoring data not available.\n\n")
            return
            
        f.write(f"## 4. Resource Utilization\n\n")
        f.write(f"### 💻 System Resources\n\n")
        f.write(f"| Resource | Mean | Max |\n")
        f.write(f"|----------|------|-----|\n")
        f.write(f"| **CPU Utilization** | {resource_data.get('cpu_utilization_mean', 0):.1f}% | {resource_data.get('cpu_utilization_max', 0):.1f}% |\n")
        f.write(f"| **Memory Utilization** | {resource_data.get('memory_utilization_mean', 0):.1f}% | {resource_data.get('memory_utilization_max', 0):.1f}% |\n")
        f.write(f"| **Memory Used** | {resource_data.get('memory_used_gb_mean', 0):.1f}GB | {resource_data.get('memory_used_gb_max', 0):.1f}GB |\n")
        
        if 'gpu_compute_utilization_mean' in resource_data:
            f.write(f"\n### 🖥️ GPU Resources\n\n")
            f.write(f"| Resource | Mean | Max |\n")
            f.write(f"|----------|------|-----|\n")
            f.write(f"| **GPU Compute** | {resource_data.get('gpu_compute_utilization_mean', 0):.1f}% | {resource_data.get('gpu_compute_utilization_max', 0):.1f}% |\n")
            f.write(f"| **GPU Memory** | {resource_data.get('gpu_memory_utilization_mean', 0):.1f}% | {resource_data.get('gpu_memory_utilization_max', 0):.1f}% |\n")
            f.write(f"| **GPU Memory Allocated** | {resource_data.get('gpu_memory_allocated_gb_mean', 0):.1f}GB | {resource_data.get('gpu_memory_allocated_gb_max', 0):.1f}GB |\n")
        
        f.write(f"\n### 📈 Monitoring Details\n\n")
        f.write(f"- **Duration**: {resource_data.get('monitoring_duration_seconds', 0):.1f}s\n")
        f.write(f"- **Total Samples**: {resource_data.get('total_samples', 0)}\n")
        f.write(f"- **Sampling Interval**: {resource_data.get('sampling_interval_seconds', 0):.1f}s\n\n")
    
    def _write_batch_analysis(self, f, experiment_results: Dict[str, Any]):
        """Write batch processing analysis"""
        batch_metrics = experiment_results.get('batch_metrics', [])
        if not batch_metrics:
            f.write(f"## 5. Batch Processing Analysis\n\n")
            f.write(f"Batch metrics not available.\n\n")
            return
            
        f.write(f"## 5. Batch Processing Analysis\n\n")
        f.write(f"### 📦 Batch Performance Overview\n\n")
        f.write(f"| Batch | Size | Duration | Input Tokens | Output Tokens | Throughput |\n")
        f.write(f"|-------|------|----------|--------------|---------------|------------|\n")
        
        for batch in batch_metrics[:10]:  # Show first 10 batches
            batch_idx = batch.get('batch_idx', 'N/A')
            batch_size = batch.get('batch_size', 0)
            duration = batch.get('batch_duration', 0)
            input_tokens = batch.get('batch_input_tokens', 0)
            output_tokens = batch.get('batch_output_tokens', 0)
            throughput = batch.get('batch_throughput_tokens_per_sec', 0)
            f.write(f"| {batch_idx} | {batch_size} | {duration:.2f}s | {input_tokens} | {output_tokens} | {throughput:.1f} |\n")
        
        if len(batch_metrics) > 10:
            f.write(f"\n*Showing first 10 of {len(batch_metrics)} batches*\n")
        
        f.write(f"\n")
    
    def _write_kv_cache_visualizations(self, f, experiment_results: Dict[str, Any], output_folder: str):
        """Write KV cache visualizations section"""
        f.write(f"## 6. KV Cache Visualizations\n\n")
        
        # Try to generate KV cache plots
        try:
            if hasattr(self, 'vllm_metrics_collector') and self.vllm_metrics_collector:
                plot_paths = self.plot_kv_cache_metrics(output_folder)
                if plot_paths:
                    f.write(f"### 📊 Cache Performance Plots\n\n")
                    for plot_type, plot_path in plot_paths.items():
                        if os.path.exists(plot_path):
                            plot_name = os.path.basename(plot_path)
                            f.write(f"#### {plot_type.replace('_', ' ').title()}\n\n")
                            f.write(f"![{plot_type}](./{plot_name})\n\n")
                            f.write(f"*Plot saved as: `{plot_name}`*\n\n")
                else:
                    f.write(f"KV cache plots could not be generated (insufficient metrics data).\n\n")
            else:
                f.write(f"KV cache visualization not available (metrics collector not initialized).\n\n")
        except Exception as e:
            f.write(f"KV cache visualization failed: {str(e)}\n\n")
    
    def _write_analysis_and_recommendations(self, f, experiment_results: Dict[str, Any]):
        """Write analysis and recommendations section"""
        f.write(f"## 7. Analysis & Recommendations\n\n")
        
        # Dataset analysis
        dataset_name = os.path.basename(experiment_results.get('experiment_info', {}).get('dataset_path', '')).lower()
        is_reordered = any(keyword in dataset_name for keyword in ['reordered', 'ggr', 'ordered', 'sorted'])
        is_shuffled = any(keyword in dataset_name for keyword in ['shuffled', 'random', 'baseline'])
        
        f.write(f"### 🔍 Dataset Analysis\n\n")
        if is_reordered:
            f.write(f"- **Dataset Type**: Optimized/Reordered\n")
            f.write(f"- **Expected Benefits**: High prefix cache hit rates due to data ordering\n")
            f.write(f"- **Use Case**: Production inference with maximum efficiency\n")
        elif is_shuffled:
            f.write(f"- **Dataset Type**: Shuffled/Baseline\n")
            f.write(f"- **Expected Benefits**: Lower hit rates, useful for comparison\n")
            f.write(f"- **Use Case**: Baseline measurements and algorithm validation\n")
        else:
            f.write(f"- **Dataset Type**: Original/Natural order\n")
            f.write(f"- **Optimization Potential**: May benefit from GGR reordering\n")
            f.write(f"- **Use Case**: Starting point for optimization experiments\n")
        
        # Performance recommendations
        f.write(f"\n### 🚀 Performance Recommendations\n\n")
        vllm_metrics = experiment_results.get('vllm_metrics', {})
        final_stats = vllm_metrics.get('final_stats', {})
        
        if final_stats.get('vllm_stats_available', False):
            key_metrics = final_stats.get('key_metrics', {})
            hit_rate = key_metrics.get('gpu_prefix_cache_hit_rate_percent', 0)
            
            if hit_rate > 50:
                f.write(f"✅ **Excellent Cache Performance** ({hit_rate:.1f}% hit rate)\n")
                f.write(f"- Current configuration is optimal\n")
                f.write(f"- Consider using this dataset ordering for production\n")
            elif hit_rate > 20:
                f.write(f"⚠️ **Moderate Cache Performance** ({hit_rate:.1f}% hit rate)\n")
                f.write(f"- Consider applying GGR reordering algorithm\n")
                f.write(f"- Experiment with different functional dependencies\n")
            else:
                f.write(f"❌ **Low Cache Performance** ({hit_rate:.1f}% hit rate)\n")
                f.write(f"- Apply GGR algorithm to reorder the dataset\n")
                f.write(f"- Verify prefix caching is properly enabled\n")
        else:
            f.write(f"⚠️ **Metrics Collection Issues**\n")
            f.write(f"- Enable vLLM metrics collection for better insights\n")
            f.write(f"- Try setting VLLM_USE_V1=0 for legacy compatibility\n")
        
        f.write(f"\n")
    
    def _write_data_exports(self, f, output_folder: str):
        """Write data exports section"""
        f.write(f"## 8. Data Exports\n\n")
        f.write(f"The following data files have been generated for further analysis:\n\n")
        
        # List expected export files
        expected_files = [
            ("experiment_results.json", "Complete experiment results in JSON format"),
            ("query_results.csv", "Individual query results and processing times"),
            ("batch_metrics.csv", "Batch-level performance metrics"),
            ("resource_metrics.csv", "System resource utilization over time"),
            ("vllm_metrics.json", "vLLM engine metrics and statistics")
        ]
        
        f.write(f"### 📁 Generated Files\n\n")
        for filename, description in expected_files:
            file_path = os.path.join(output_folder, filename)
            if os.path.exists(file_path):
                f.write(f"✅ **{filename}**: {description}\n")
            else:
                f.write(f"⚠️ **{filename}**: {description} *(file not found)*\n")
        
        f.write(f"\n### 🔧 Usage Instructions\n\n")
        f.write(f"```bash\n")
        f.write(f"# View JSON results\n")
        f.write(f"jq . {output_folder}/experiment_results.json\n\n")
        f.write(f"# Analyze CSV data with pandas\n")
        f.write(f"python -c \"import pandas as pd; df = pd.read_csv('{output_folder}/batch_metrics.csv'); print(df.describe())\"\n")
        f.write(f"```\n\n")
        
        f.write(f"---\n\n")
        f.write(f"**Report Generated**: {datetime.now().isoformat()}  \n")
        f.write(f"**Tool**: vLLM Experiment Pipeline with GGR Algorithm  \n")
        f.write(f"**Version**: Enhanced Metrics Collection  \n")
