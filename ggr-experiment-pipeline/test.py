#!/usr/bin/env python3
"""
Test script for GGR Experiment Pipeline
Creates a sample dataset and runs a basic experiment
"""
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def create_sample_dataset():
    """Create a sample dataset for testing"""
    np.random.seed(42)
    
    # Create sample movie data
    movie_ids = [f"movie_{i:03d}" for i in range(1, 51)]
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']
    
    data = []
    for movie_id in movie_ids:
        # Each movie has a fixed title and genre (functional dependency)
        title = f"Movie Title {movie_id.split('_')[1]}"
        genre = np.random.choice(genres)
        
        # Generate multiple reviews per movie
        num_reviews = np.random.randint(2, 6)
        for _ in range(num_reviews):
            review = f"This is a review for {title}. " + \
                    f"{'Great movie!' if np.random.random() > 0.3 else 'Not recommended.'}"
            critic = f"Critic_{np.random.randint(1, 21):02d}"
            rating = np.random.randint(1, 11)
            
            data.append({
                'movie_id': movie_id,
                'movie_title': title,
                'genre': genre,
                'review_content': review,
                'critic_name': critic,
                'rating': rating
            })
    
    df = pd.DataFrame(data)
    return df

def run_test():
    """Run a basic test of the GGR experiment pipeline"""
    print("Creating sample dataset...")
    
    # Create sample data
    df = create_sample_dataset()
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    dataset_path = 'data/sample_movies.csv'
    df.to_csv(dataset_path, index=False)
    
    print(f"Sample dataset created: {dataset_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Test the experiment pipeline
    print("\n" + "="*60)
    print("RUNNING TEST EXPERIMENT")
    print("="*60)
    
    try:
        from src.experiment_runner import run_single_experiment
        
        # Run experiment with known functional dependencies
        results = run_single_experiment(
            dataset_path=dataset_path,
            functional_dependencies="movie_id->movie_title,movie_id->genre",
            columns="movie_id,movie_title,genre,critic_name",
            output_dir="test_results",
            experiment_name="test_experiment",
            max_depth=20,  # Limit depth for testing
            discover_fds=False,  # Use provided FDs
            handle_missing='drop'
        )
        
        print("TEST PASSED!")
        print(f"Total hits: {results['total_hits']}")
        print(f"Execution time: {results['total_execution_time']:.2f} seconds")
        
        if 'output_files' in results:
            print("\nGenerated files:")
            for file_type, path in results['output_files'].items():
                print(f"  {file_type}: {path}")
                
        return True
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
