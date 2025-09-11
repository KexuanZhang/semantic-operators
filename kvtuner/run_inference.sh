#!/bin/bash

# KVTuner Inference Script Runner for Qwen2.5-3B-Instruct
# This script provides convenient ways to run KVTuner inference

# Configuration
MODEL_PATH="/Users/zhang/Desktop/huawei/untitled folder 6/Qwen2.5-3B-Instruct"
KVTUNER_DIR="/Users/zhang/Desktop/huawei/untitled folder 6/KVTuner"
DATASET_DIR="/Users/zhang/Desktop/huawei/untitled folder 6/semantic-operators/datasets"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if model path exists
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "Model path not found: $MODEL_PATH"
        print_info "Please update MODEL_PATH in this script to point to your Qwen2.5-3B-Instruct model"
        exit 1
    fi
    
    # Check if KVTuner directory exists
    if [ ! -d "$KVTUNER_DIR" ]; then
        print_error "KVTuner directory not found: $KVTUNER_DIR"
        print_info "Please update KVTUNER_DIR in this script to point to your KVTuner installation"
        exit 1
    fi
    
    # Check if Python scripts exist
    if [ ! -f "kvtuner_inference.py" ]; then
        print_error "kvtuner_inference.py not found in current directory"
        exit 1
    fi
    
    if [ ! -f "kvtuner_simple.py" ]; then
        print_error "kvtuner_simple.py not found in current directory"
        exit 1
    fi
    
    print_success "Prerequisites check passed!"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  simple-test     Run simple test with a few predefined prompts"
    echo "  simple-dataset  Run simple inference on a dataset"
    echo "  full-dataset    Run full inference with comprehensive options"
    echo "  kivi-test       Run test with KiVi quantization scheme"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 simple-test"
    echo "  $0 simple-dataset"
    echo "  $0 full-dataset"
    echo ""
    echo "Configuration (edit this script to change):"
    echo "  Model path: $MODEL_PATH"
    echo "  KVTuner path: $KVTUNER_DIR"
    echo "  Dataset dir: $DATASET_DIR"
}

# Function to run simple test
run_simple_test() {
    print_info "Running simple test with KVTuner (pertoken scheme)..."
    python kvtuner_simple.py \
        --model_path "$MODEL_PATH" \
        --scheme pertoken \
        --test
}

# Function to run simple dataset inference
run_simple_dataset() {
    # Check for dataset files
    if [ ! -d "$DATASET_DIR" ]; then
        print_warning "Dataset directory not found: $DATASET_DIR"
        print_info "Please provide a dataset file path:"
        read -p "Enter dataset CSV file path: " dataset_path
    else
        # Look for CSV files in dataset directory
        csv_files=($(find "$DATASET_DIR" -name "*.csv" | head -5))
        
        if [ ${#csv_files[@]} -eq 0 ]; then
            print_warning "No CSV files found in $DATASET_DIR"
            read -p "Enter dataset CSV file path: " dataset_path
        else
            print_info "Available CSV files:"
            for i in "${!csv_files[@]}"; do
                echo "  $((i+1)). ${csv_files[$i]}"
            done
            echo "  0. Enter custom path"
            
            read -p "Select a dataset (1-${#csv_files[@]}, or 0 for custom): " choice
            
            if [ "$choice" -eq 0 ]; then
                read -p "Enter dataset CSV file path: " dataset_path
            elif [ "$choice" -ge 1 ] && [ "$choice" -le ${#csv_files[@]} ]; then
                dataset_path="${csv_files[$((choice-1))]}"
            else
                print_error "Invalid selection"
                return 1
            fi
        fi
    fi
    
    if [ ! -f "$dataset_path" ]; then
        print_error "Dataset file not found: $dataset_path"
        return 1
    fi
    
    print_info "Running simple dataset inference..."
    print_info "Dataset: $dataset_path"
    
    python kvtuner_simple.py \
        --model_path "$MODEL_PATH" \
        --dataset "$dataset_path" \
        --scheme pertoken \
        --max_rows 10 \
        --max_new_tokens 256
}

# Function to run full dataset inference
run_full_dataset() {
    # Get dataset path
    if [ ! -d "$DATASET_DIR" ]; then
        print_warning "Dataset directory not found: $DATASET_DIR"
        read -p "Enter dataset CSV file path: " dataset_path
    else
        csv_files=($(find "$DATASET_DIR" -name "*.csv" | head -5))
        
        if [ ${#csv_files[@]} -eq 0 ]; then
            print_warning "No CSV files found in $DATASET_DIR"
            read -p "Enter dataset CSV file path: " dataset_path
        else
            print_info "Available CSV files:"
            for i in "${!csv_files[@]}"; do
                echo "  $((i+1)). ${csv_files[$i]}"
            done
            echo "  0. Enter custom path"
            
            read -p "Select a dataset (1-${#csv_files[@]}, or 0 for custom): " choice
            
            if [ "$choice" -eq 0 ]; then
                read -p "Enter dataset CSV file path: " dataset_path
            elif [ "$choice" -ge 1 ] && [ "$choice" -le ${#csv_files[@]} ]; then
                dataset_path="${csv_files[$((choice-1))]}"
            else
                print_error "Invalid selection"
                return 1
            fi
        fi
    fi
    
    if [ ! -f "$dataset_path" ]; then
        print_error "Dataset file not found: $dataset_path"
        return 1
    fi
    
    # Get quantization scheme
    echo "Select quantization scheme:"
    echo "  1. pertoken (default)"
    echo "  2. kivi"
    read -p "Choice (1-2): " scheme_choice
    
    case $scheme_choice in
        1|"") scheme="pertoken" ;;
        2) scheme="kivi" ;;
        *) 
            print_error "Invalid choice"
            return 1
            ;;
    esac
    
    # Get other parameters
    read -p "Max rows to process (default: all): " max_rows
    read -p "Max new tokens (default: 256): " max_tokens
    read -p "Text column name (default: text): " text_column
    
    # Set defaults
    max_rows=${max_rows:-""}
    max_tokens=${max_tokens:-256}
    text_column=${text_column:-"text"}
    
    print_info "Running full dataset inference..."
    print_info "Dataset: $dataset_path"
    print_info "Scheme: $scheme"
    print_info "Max tokens: $max_tokens"
    print_info "Text column: $text_column"
    
    # Build command
    cmd="python kvtuner_inference.py --model_path \"$MODEL_PATH\" --dataset \"$dataset_path\" --scheme $scheme --max_new_tokens $max_tokens --text_column \"$text_column\""
    
    if [ -n "$max_rows" ]; then
        cmd="$cmd --max_rows $max_rows"
    fi
    
    print_info "Running: $cmd"
    eval $cmd
}

# Function to run KiVi test
run_kivi_test() {
    print_info "Running simple test with KiVi quantization scheme..."
    python kvtuner_simple.py \
        --model_path "$MODEL_PATH" \
        --scheme kivi \
        --test
}

# Main script logic
main() {
    check_prerequisites
    
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    case $1 in
        "simple-test")
            run_simple_test
            ;;
        "simple-dataset")
            run_simple_dataset
            ;;
        "full-dataset")
            run_full_dataset
            ;;
        "kivi-test")
            run_kivi_test
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
