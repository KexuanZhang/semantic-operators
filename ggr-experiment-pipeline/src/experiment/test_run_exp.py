#!/usr/bin/env python3
"""
Test script for the sentiment analysis experiment runner.
Tests the core functionality without requiring an actual LLM model.
"""

import unittest
import tempfile
import os
import pandas as pd
import json
from pathlib import Path
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_exp import SentimentAnalysisExperiment

class MockLLM:
    """Mock LLM for testing purposes."""
    
    def generate(self, prompts, sampling_params):
        """Mock generate method that returns fake sentiment predictions."""
        class MockOutput:
            def __init__(self, text):
                self.text = text
        
        class MockResult:
            def __init__(self, text):
                self.outputs = [MockOutput(text)]
        
        # Simple mock logic: if "love" or "great" in prompt, return POSITIVE
        prompt = prompts[0].lower()
        if any(word in prompt for word in ["love", "great", "amazing", "wonderful", "excellent"]):
            return [MockResult("POSITIVE")]
        elif any(word in prompt for word in ["hate", "terrible", "awful", "bad", "worst"]):
            return [MockResult("NEGATIVE")]
        else:
            return [MockResult("POSITIVE")]  # Default


class TestSentimentAnalysisExperiment(unittest.TestCase):
    """Test cases for the SentimentAnalysisExperiment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = "/mock/model/path"
        self.experiment = SentimentAnalysisExperiment(
            model_path=self.model_path,
            results_dir=self.temp_dir
        )
        
        # Create sample dataset
        self.sample_data = [
            {"text": "I love this product!", "label": "positive"},
            {"text": "This is terrible.", "label": "negative"}, 
            {"text": "Great service!", "label": "positive"},
            {"text": "Awful experience.", "label": "negative"},
        ]
        
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset.csv")
        df = pd.DataFrame(self.sample_data)
        df.to_csv(self.dataset_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test experiment initialization."""
        self.assertEqual(self.experiment.model_path, self.model_path)
        self.assertTrue(self.experiment.results_dir.exists())
        self.assertIsNone(self.experiment.llm)
    
    def test_create_sentiment_prompt(self):
        """Test sentiment prompt creation."""
        text = "This is a test"
        prompt = self.experiment.create_sentiment_prompt(text)
        
        self.assertIn(text, prompt)
        self.assertIn("POSITIVE", prompt)
        self.assertIn("NEGATIVE", prompt)
        self.assertIn("Sentiment:", prompt)
    
    def test_analyze_sentiment_mock(self):
        """Test sentiment analysis with mock LLM."""
        # Replace the LLM with mock for testing
        self.experiment.llm = MockLLM()
        
        # Test positive sentiment
        positive_result = self.experiment.analyze_sentiment("I love this!")
        self.assertEqual(positive_result, "POSITIVE")
        
        # Test negative sentiment
        negative_result = self.experiment.analyze_sentiment("This is terrible!")
        self.assertEqual(negative_result, "NEGATIVE")
    
    def test_analyze_sentiment_no_model(self):
        """Test sentiment analysis without loaded model."""
        with self.assertRaises(ValueError):
            self.experiment.analyze_sentiment("test text")
    
    def test_process_dataset_mock(self):
        """Test dataset processing with mock LLM."""
        # Replace the LLM with mock for testing
        self.experiment.llm = MockLLM()
        
        result = self.experiment.process_dataset(
            dataset_path=self.dataset_path,
            text_column="text"
        )
        
        # Verify result structure
        self.assertIn("results", result)
        self.assertIn("metadata", result)
        
        # Verify results
        results = result["results"]
        self.assertEqual(len(results), 4)
        
        # Check first result structure
        first_result = results[0]
        required_keys = ["index", "text", "predicted_sentiment", "inference_time", "timestamp"]
        for key in required_keys:
            self.assertIn(key, first_result)
        
        # Verify metadata
        metadata = result["metadata"]
        self.assertEqual(metadata["total_samples"], 4)
        self.assertEqual(metadata["processed_samples"], 4)
        self.assertEqual(metadata["model_path"], self.model_path)
        self.assertEqual(metadata["text_column"], "text")
    
    def test_process_dataset_invalid_column(self):
        """Test dataset processing with invalid column name."""
        self.experiment.llm = MockLLM()
        
        with self.assertRaises(ValueError):
            self.experiment.process_dataset(
                dataset_path=self.dataset_path,
                text_column="nonexistent_column"
            )
    
    def test_save_results(self):
        """Test saving experiment results."""
        # Create mock experiment data
        experiment_data = {
            "results": [
                {
                    "index": 0,
                    "text": "test text",
                    "predicted_sentiment": "POSITIVE",
                    "inference_time": 0.1,
                    "timestamp": "2024-01-01T00:00:00"
                }
            ],
            "metadata": {
                "total_samples": 1,
                "positive_predictions": 1,
                "negative_predictions": 0,
                "total_inference_time_seconds": 0.1
            }
        }
        
        experiment_name = "test_experiment"
        self.experiment.save_results(experiment_data, experiment_name)
        
        # Verify files were created
        results_file = self.experiment.results_dir / f"{experiment_name}_results.json"
        csv_file = self.experiment.results_dir / f"{experiment_name}_results.csv"
        metadata_file = self.experiment.results_dir / f"{experiment_name}_metadata.json"
        
        self.assertTrue(results_file.exists())
        self.assertTrue(csv_file.exists())
        self.assertTrue(metadata_file.exists())
        
        # Verify JSON content
        with open(results_file, 'r') as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data, experiment_data)
        
        # Verify CSV content
        df = pd.read_csv(csv_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['predicted_sentiment'], 'POSITIVE')


def run_integration_test():
    """
    Run an integration test that simulates the full workflow.
    This test uses mock objects to avoid requiring an actual model.
    """
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TEST")
    print("="*60)
    
    # Create temporary directory and dataset
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test dataset
        sample_data = [
            {"text": "I absolutely love this product! It's amazing!", "true_label": "positive"},
            {"text": "This is the worst thing I've ever bought.", "true_label": "negative"},
            {"text": "Great quality and fast shipping. Highly recommend!", "true_label": "positive"},
            {"text": "Terrible customer service and poor quality.", "true_label": "negative"},
            {"text": "Excellent value for money. Very satisfied!", "true_label": "positive"},
        ]
        
        dataset_path = os.path.join(temp_dir, "integration_test_dataset.csv")
        df = pd.DataFrame(sample_data)
        df.to_csv(dataset_path, index=False)
        print(f"Created test dataset: {dataset_path}")
        
        # Initialize experiment
        experiment = SentimentAnalysisExperiment(
            model_path="/mock/model/path",
            results_dir=temp_dir
        )
        
        # Mock the LLM
        experiment.llm = MockLLM()
        print("Initialized experiment with mock LLM")
        
        # Process dataset
        print("Processing dataset...")
        results = experiment.process_dataset(
            dataset_path=dataset_path,
            text_column="text"
        )
        
        # Save results
        experiment.save_results(results, "integration_test")
        print("Results saved successfully")
        
        # Print summary
        metadata = results["metadata"]
        print(f"\nIntegration Test Summary:")
        print(f"- Total samples processed: {metadata['total_samples']}")
        print(f"- Positive predictions: {metadata['positive_predictions']}")
        print(f"- Negative predictions: {metadata['negative_predictions']}")
        print(f"- Total time: {metadata['total_inference_time_seconds']:.4f} seconds")
        print(f"- Average time per sample: {metadata['average_inference_time_seconds']:.4f} seconds")
        
        # Verify prediction accuracy (with mock LLM)
        correct_predictions = 0
        for i, result in enumerate(results["results"]):
            predicted = result["predicted_sentiment"].lower()
            true_label = sample_data[i]["true_label"]
            if predicted == true_label:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(sample_data)
        print(f"- Mock prediction accuracy: {accuracy:.2%}")
        
        print("\n✅ Integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    print("Running Sentiment Analysis Experiment Tests")
    print("=" * 60)
    
    # Run unit tests
    print("\n1. Running unit tests...")
    unittest.main(argv=[''], verbosity=2, exit=False)
    
    # Run integration test
    print("\n2. Running integration test...")
    run_integration_test()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
