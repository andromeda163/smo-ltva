#!/bin/bash
#
# Run all tracking comparisons and summarize results
#

set -e

OUTPUT_DIR="outputs/trajectories"
SUMMARY_FILE="$OUTPUT_DIR/tracking_summary.txt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clear previous summary
> "$SUMMARY_FILE"

echo "========================================"
echo "Point Tracking Comparison Script"
echo "========================================"
echo ""
echo "Running comparisons for all test videos..."
echo ""

# Function to run comparison and capture stats
run_comparison() {
    local video=$1
    local points=$2
    local name=$(basename "$video")
    
    echo "----------------------------------------" | tee -a "$SUMMARY_FILE"
    echo "Video: $name" | tee -a "$SUMMARY_FILE"
    echo "Initial points: $points" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
    
    # Run Python comparison and capture output
    python3 point_tracker.py --video "$video" --points $points --comparison 2>&1 | tee -a "$SUMMARY_FILE"
    
    echo "" | tee -a "$SUMMARY_FILE"
}

# Run comparisons for each video
run_comparison "data/bear01" 100
run_comparison "data/cars2" 50
run_comparison "data/horses03" 150

echo "========================================" | tee -a "$SUMMARY_FILE"
echo "Summary Complete"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.mp4 2>/dev/null | awk '{print "  " $NF " (" int($5/1024/1024) " MB)"}' | tee -a "$SUMMARY_FILE"
echo ""
echo "Summary saved to: $SUMMARY_FILE"
