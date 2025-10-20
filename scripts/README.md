# 🚀 Fine-Tuning Management Scripts

## Quick Start

Use `manage_fine_tuning.py` for all fine-tuning operations:

### Start a New Fine-Tuning Job
```bash
python scripts/manage_fine_tuning.py start
```

### Check Status of All Jobs
```bash
python scripts/manage_fine_tuning.py status
```

### Check Status of Specific Job
```bash
python scripts/manage_fine_tuning.py status ftjob-xxxxx
```

### Merge Daily Files into One (One-Time)
```bash
python scripts/manage_fine_tuning.py merge
```

### Clean Up Old Training Files
```bash
python scripts/manage_fine_tuning.py cleanup
```

---

## What Each Command Does

### `start` - Start Fine-Tuning
1. ✅ Loads all corrections from `data/fine_tuning_dataset/corrections.jsonl`
2. ✅ Checks you have at least 10 corrections (minimum recommended)
3. ✅ Converts to OpenAI's fine-tuning format
4. ✅ Uploads to OpenAI
5. ✅ Starts fine-tuning job with `gpt-4o-mini`
6. ✅ Saves job ID to `data/fine_tuning_status.json`
7. ✅ Auto-cleans up old training files

### `merge` - Merge Legacy Files
- Consolidates old daily `corrections_YYYY-MM-DD.jsonl` files into single `corrections.jsonl`
- Detects and skips duplicates
- Optionally deletes legacy files after merge
- **One-time operation** - only needed if you have old daily files

### `status` - Check Job Status
- Shows all your fine-tuning jobs (or specific job if ID provided)
- Displays current status: queued, running, succeeded, or failed
- Shows fine-tuned model name when complete
- Filters to show only `gpt-4o-mini` jobs

### `cleanup` - Clean Old Files
- Finds all `training_data_*.jsonl` files
- Shows file sizes
- Asks for confirmation before deleting
- Useful to save disk space

---

## Typical Workflow

1. **Collect Feedback** 
   - Users click 👎 on bad responses
   - Auto-corrections saved to `corrections_*.jsonl`

2. **Start Fine-Tuning** (when you have 10+ corrections)
   ```bash
   python scripts/manage_fine_tuning.py start
   ```

3. **Monitor Progress**
   ```bash
   python scripts/manage_fine_tuning.py status
   ```

4. **Use Fine-Tuned Model** (when status = succeeded)
   - Update `model_name` in `app/endpoints.py`
   - Replace `"gpt-4o-mini"` with your fine-tuned model ID

---

## Files Created

- `data/fine_tuning_dataset/corrections.jsonl` - **Single unified file** for all auto-corrections (keep this!)
- `data/fine_tuning_dataset/training_data_*.jsonl` - Converted format (deleted after upload)
- `data/fine_tuning_status.json` - Latest job info

---

## Tips

✅ **Minimum 10 corrections** - More is better (50+ ideal)  
✅ **Wait 20 mins - 2 hours** - Fine-tuning takes time  
✅ **Check OpenAI dashboard** - https://platform.openai.com/finetune  
✅ **Keep corrections.jsonl** - This is your source data (backed up automatically)  
✅ **Clean up regularly** - Use `cleanup` command to remove old training files  
✅ **Single file is better** - All corrections now append to one file (simpler!)  

---

## Troubleshooting

**"No corrections found"**
- Users need to give thumbs down feedback first
- Check `data/fine_tuning_dataset/corrections_*.jsonl` exists

**"Only X corrections found"**
- You can continue, but 10+ is recommended
- Collect more feedback for better results

**"Job failed"**
- Check error message in status output
- Common issues: invalid data format, insufficient training examples
- Contact OpenAI support if needed

---

## Advanced Usage

### Check Specific Job by ID
```bash
python scripts/manage_fine_tuning.py status ftjob-abc123xyz
```

### View Help
```bash
python scripts/manage_fine_tuning.py -h
```

---

**Need help?** Check the main documentation or reach out to the team!

