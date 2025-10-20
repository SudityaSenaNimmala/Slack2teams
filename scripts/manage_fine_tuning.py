#!/usr/bin/env python3
"""
Fine-Tuning Management Script for CloudFuze Chatbot
Unified tool to start, monitor, and manage OpenAI fine-tuning jobs.

Usage:
    python scripts/manage_fine_tuning.py start     # Start new fine-tuning job
    python scripts/manage_fine_tuning.py status    # Check status of jobs
    python scripts/manage_fine_tuning.py cleanup   # Clean up old training files
"""

import json
import os
import glob
import argparse
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ======================== DATA LOADING ========================

def load_correction_dataset():
    """Load all correction data from JSONL files (supports both single file and legacy daily files)."""
    dataset_dir = "./data/fine_tuning_dataset"
    all_corrections = []
    
    if not os.path.exists(dataset_dir):
        print("[ERROR] No fine-tuning dataset found!")
        return []
    
    # Check for new single file format first
    single_file = f"{dataset_dir}/corrections.jsonl"
    if os.path.exists(single_file):
        print(f"[*] Loading from unified corrections file")
        with open(single_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        correction = json.loads(line.strip())
                        all_corrections.append(correction)
                    except json.JSONDecodeError as e:
                        print(f"[WARNING] Error parsing line: {e}")
        print(f"[OK] Loaded {len(all_corrections)} corrections from unified file")
    
    # Also load legacy daily files if they exist
    legacy_files = glob.glob(f"{dataset_dir}/corrections_*.jsonl")
    if legacy_files:
        print(f"[*] Found {len(legacy_files)} legacy daily correction files")
        print(f"[TIP] Run 'python scripts/manage_fine_tuning.py merge' to consolidate them")
        
        for file_path in legacy_files:
            print(f"[*] Loading: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            correction = json.loads(line.strip())
                            all_corrections.append(correction)
                        except json.JSONDecodeError as e:
                            print(f"[WARNING] Error parsing line: {e}")
    
    if not all_corrections:
        print("[ERROR] No correction files found!")
        return []
    
    print(f"[OK] Total loaded: {len(all_corrections)} corrections")
    return all_corrections

def prepare_training_data(corrections):
    """Prepare data in OpenAI fine-tuning format."""
    training_data = []
    
    for correction in corrections:
        # Create training example in chat format
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert assistant specializing in Slack to Microsoft Teams migrations via CloudFuze. Provide accurate, helpful, and specific information about migration processes, tools, and best practices."
                },
                {
                    "role": "user", 
                    "content": correction["input"]
                },
                {
                    "role": "assistant",
                    "content": correction["corrected_output"]
                }
            ]
        }
        training_data.append(training_example)
    
    return training_data

# ======================== FINE-TUNING ========================

def start_fine_tuning(training_data):
    """Start OpenAI fine-tuning job."""
    try:
        client = OpenAI()
        
        # Save training data to file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        training_file = f"./data/fine_tuning_dataset/training_data_{timestamp}.jsonl"
        
        with open(training_file, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"[*] Training data saved to: {training_file}")
        
        # Upload training file to OpenAI
        print("[*] Uploading training file to OpenAI...")
        with open(training_file, 'rb') as f:
            upload_response = client.files.create(
                file=f,
                purpose="fine-tune"
            )
        
        print(f"[OK] File uploaded: {upload_response.id}")
        
        # Start fine-tuning job
        print("[*] Starting fine-tuning job...")
        fine_tune_response = client.fine_tuning.jobs.create(
            training_file=upload_response.id,
            model="gpt-4o-mini-2024-07-18"  # Using gpt-4o-mini as base model
        )
        
        print(f"[OK] Fine-tuning job started!")
        print(f"   Job ID: {fine_tune_response.id}")
        print(f"   Status: {fine_tune_response.status}")
        
        # Save job info
        job_info = {
            "job_id": fine_tune_response.id,
            "status": fine_tune_response.status,
            "model": "gpt-4o-mini-2024-07-18",
            "training_file": upload_response.id,
            "created_at": datetime.now().isoformat(),
            "corrections_count": len(training_data)
        }
        
        with open("./data/fine_tuning_status.json", 'w', encoding='utf-8') as f:
            json.dump(job_info, f, indent=2, ensure_ascii=False)
        
        # Clean up old training files after successful upload
        cleanup_old_training_files(training_file)
        
        return fine_tune_response.id
        
    except Exception as e:
        print(f"[ERROR] Error starting fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return None

# ======================== STATUS CHECKING ========================

def check_fine_tuning_status(job_id=None):
    """Check the status of fine-tuning jobs."""
    try:
        client = OpenAI()
        
        print("\n" + "=" * 70)
        print("FINE-TUNING JOBS STATUS")
        print("=" * 70)
        
        if job_id:
            # Check specific job
            try:
                job = client.fine_tuning.jobs.retrieve(job_id)
                display_job_status(job)
            except Exception as e:
                print(f"[ERROR] Could not retrieve job {job_id}: {e}")
        else:
            # Get all fine-tuning jobs
            jobs = client.fine_tuning.jobs.list(limit=10)
            
            if not jobs.data:
                print("\n[*] No fine-tuning jobs found")
                return
            
            # Filter to show only gpt-4o-mini jobs
            gpt4o_jobs = [job for job in jobs.data if "gpt-4o-mini" in job.model]
            
            if not gpt4o_jobs:
                print("\n[*] No gpt-4o-mini fine-tuning jobs found")
                return
            
            print(f"\n[*] Found {len(gpt4o_jobs)} gpt-4o-mini fine-tuning job(s)\n")
            
            for idx, job in enumerate(gpt4o_jobs, 1):
                print(f"\n--- Job {idx} ---")
                display_job_status(job)
        
        # Check local status file
        display_local_status()
        
    except Exception as e:
        print(f"[ERROR] Error checking fine-tuning status: {e}")
        import traceback
        traceback.print_exc()

def display_job_status(job):
    """Display formatted job status."""
    print(f"\nJob ID: {job.id}")
    print(f"Model: {job.model}")
    print(f"Status: {job.status}")
    print(f"Created: {datetime.fromtimestamp(job.created_at).strftime('%Y-%m-%d %H:%M:%S')}")
    
    if job.fine_tuned_model:
        print(f"Fine-tuned Model: {job.fine_tuned_model}")
    
    # Status-specific messages
    if job.status == "succeeded":
        print(f"\n[SUCCESS] Job completed successfully!")
        print(f"   Your fine-tuned model: {job.fine_tuned_model}")
        print(f"\n   To use this model, update app/endpoints.py:")
        print(f"   model_name=\"{job.fine_tuned_model}\"")
    elif job.status == "failed":
        print(f"\n[FAILED] Job failed")
        if hasattr(job, 'error') and job.error:
            print(f"   Error: {job.error}")
    elif job.status == "running":
        print(f"\n[IN PROGRESS] Job is running...")
        print(f"   This typically takes 20 minutes to 2 hours")
    elif job.status == "validating_files":
        print(f"\n[VALIDATING] Validating training files...")
    elif job.status == "queued":
        print(f"\n[QUEUED] Job is queued and will start soon...")

def display_local_status():
    """Display local status file information."""
    if os.path.exists("./data/fine_tuning_status.json"):
        try:
            with open("./data/fine_tuning_status.json", 'r', encoding='utf-8') as f:
                local_status = json.load(f)
            
            print(f"\n" + "-" * 70)
            print("LOCAL STATUS FILE:")
            print("-" * 70)
            print(f"Job ID: {local_status.get('job_id', 'N/A')}")
            print(f"Corrections Used: {local_status.get('corrections_count', 'N/A')}")
            print(f"Created: {local_status.get('created_at', 'N/A')}")
        except Exception as e:
            print(f"[WARNING] Could not read local status file: {e}")

# ======================== CLEANUP ========================

def cleanup_old_training_files(keep_file=None):
    """Clean up old training_data_*.jsonl files."""
    try:
        old_files = glob.glob("./data/fine_tuning_dataset/training_data_*.jsonl")
        deleted_count = 0
        
        for old_file in old_files:
            if keep_file and old_file == keep_file:
                continue  # Don't delete the current file
            
            try:
                os.remove(old_file)
                print(f"[OK] Deleted old training file: {os.path.basename(old_file)}")
                deleted_count += 1
            except Exception as e:
                print(f"[WARNING] Could not delete {os.path.basename(old_file)}: {e}")
        
        if deleted_count > 0:
            print(f"[OK] Cleaned up {deleted_count} old training file(s)")
        
    except Exception as e:
        print(f"[WARNING] Cleanup failed: {e}")

def cleanup_command():
    """Manual cleanup of old training files."""
    print("\n" + "=" * 70)
    print("CLEANUP OLD TRAINING FILES")
    print("=" * 70)
    
    old_files = glob.glob("./data/fine_tuning_dataset/training_data_*.jsonl")
    
    if not old_files:
        print("\n[*] No training files to clean up")
        return
    
    print(f"\n[*] Found {len(old_files)} training file(s):")
    for f in old_files:
        size = os.path.getsize(f) / 1024  # KB
        print(f"   - {os.path.basename(f)} ({size:.1f} KB)")
    
    response = input("\nDelete all these files? (y/n): ")
    if response.lower() == 'y':
        cleanup_old_training_files()
        print("\n[SUCCESS] Cleanup complete!")
    else:
        print("\n[*] Cleanup cancelled")

def merge_command():
    """Merge legacy daily correction files into single unified file."""
    print("\n" + "=" * 70)
    print("MERGE CORRECTION FILES")
    print("=" * 70)
    
    dataset_dir = "./data/fine_tuning_dataset"
    single_file = f"{dataset_dir}/corrections.jsonl"
    legacy_files = glob.glob(f"{dataset_dir}/corrections_*.jsonl")
    
    if not legacy_files:
        print("\n[*] No legacy daily files to merge")
        return
    
    print(f"\n[*] Found {len(legacy_files)} legacy daily file(s):")
    total_size = 0
    for f in legacy_files:
        size = os.path.getsize(f) / 1024  # KB
        total_size += size
        print(f"   - {os.path.basename(f)} ({size:.1f} KB)")
    
    print(f"\n[*] Total size: {total_size:.1f} KB")
    
    # Count existing corrections in unified file
    existing_count = 0
    existing_trace_ids = set()
    if os.path.exists(single_file):
        with open(single_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        correction = json.loads(line.strip())
                        existing_count += 1
                        existing_trace_ids.add(correction.get('trace_id'))
                    except json.JSONDecodeError:
                        pass
        print(f"\n[*] Unified file already has {existing_count} corrections")
    
    # Load all legacy corrections
    all_corrections = []
    duplicates = 0
    for file_path in legacy_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        correction = json.loads(line.strip())
                        # Skip if already in unified file
                        if correction.get('trace_id') in existing_trace_ids:
                            duplicates += 1
                        else:
                            all_corrections.append(correction)
                    except json.JSONDecodeError as e:
                        print(f"[WARNING] Error parsing line: {e}")
    
    if not all_corrections:
        print(f"\n[*] All {duplicates} corrections already in unified file")
        print(f"[*] Safe to delete legacy files")
    else:
        print(f"\n[*] Will merge {len(all_corrections)} new corrections")
        if duplicates > 0:
            print(f"[*] Skipping {duplicates} duplicates")
    
    response = input("\nProceed with merge? (y/n): ")
    if response.lower() != 'y':
        print("\n[*] Merge cancelled")
        return
    
    # Append to unified file
    if all_corrections:
        with open(single_file, 'a', encoding='utf-8') as f:
            for correction in all_corrections:
                f.write(json.dumps(correction, ensure_ascii=False) + '\n')
        print(f"\n[OK] Merged {len(all_corrections)} corrections to {os.path.basename(single_file)}")
    
    # Ask to delete legacy files
    response = input("\nDelete legacy daily files? (y/n): ")
    if response.lower() == 'y':
        for file_path in legacy_files:
            os.remove(file_path)
            print(f"[OK] Deleted {os.path.basename(file_path)}")
        print(f"\n[SUCCESS] Merge complete! Deleted {len(legacy_files)} legacy file(s)")
    else:
        print("\n[*] Legacy files kept (you can delete them manually later)")
    
    # Show final stats
    final_count = existing_count + len(all_corrections)
    print(f"\n[SUMMARY]")
    print(f"   Unified file: {os.path.basename(single_file)}")
    print(f"   Total corrections: {final_count}")
    print(f"   File size: {os.path.getsize(single_file) / 1024:.1f} KB")

# ======================== MAIN WORKFLOW ========================

def start_command():
    """Start new fine-tuning job."""
    print("\n" + "=" * 70)
    print("START FINE-TUNING JOB")
    print("=" * 70)
    
    # Step 1: Load correction dataset
    print("\n[STEP 1] Loading correction dataset...")
    corrections = load_correction_dataset()
    
    if not corrections:
        print("\n[ERROR] No corrections found. Collect more feedback first.")
        print("   Users need to click thumbs down (👎) on responses to generate corrections.")
        return
    
    # Step 2: Check data quality
    print(f"\n[STEP 2] Data quality check...")
    print(f"   Total corrections: {len(corrections)}")
    
    # Check for minimum samples
    if len(corrections) < 10:
        print(f"\n[WARNING] Only {len(corrections)} corrections found")
        print(f"   Recommended minimum: 10 corrections")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\n[*] Fine-tuning cancelled")
            return
    
    # Step 3: Prepare training data
    print(f"\n[STEP 3] Preparing training data...")
    training_data = prepare_training_data(corrections)
    print(f"[OK] Prepared {len(training_data)} training examples")
    
    # Step 4: Start fine-tuning
    print(f"\n[STEP 4] Starting fine-tuning job...")
    job_id = start_fine_tuning(training_data)
    
    if job_id:
        print(f"\n" + "=" * 70)
        print("[SUCCESS] Fine-tuning job started!")
        print("=" * 70)
        print(f"\nJob ID: {job_id}")
        print(f"\nNext steps:")
        print(f"  1. Monitor: https://platform.openai.com/finetune")
        print(f"  2. Check status: python scripts/manage_fine_tuning.py status")
        print(f"  3. Wait 20 mins - 2 hours for completion")
        print("=" * 70)
    else:
        print(f"\n[ERROR] Failed to start fine-tuning job")

# ======================== CLI ========================

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="CloudFuze Chatbot Fine-Tuning Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/manage_fine_tuning.py start          # Start new fine-tuning job
  python scripts/manage_fine_tuning.py status         # Check all jobs
  python scripts/manage_fine_tuning.py status JOB_ID  # Check specific job
  python scripts/manage_fine_tuning.py merge          # Merge daily files into one
  python scripts/manage_fine_tuning.py cleanup        # Clean up old training files
        """
    )
    
    parser.add_argument(
        'command',
        choices=['start', 'status', 'merge', 'cleanup'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'job_id',
        nargs='?',
        help='Job ID to check (optional, for status command)'
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'start':
        start_command()
    elif args.command == 'status':
        check_fine_tuning_status(args.job_id)
    elif args.command == 'merge':
        merge_command()
    elif args.command == 'cleanup':
        cleanup_command()

if __name__ == "__main__":
    main()

