# Pipeline Failure Scenarios - What-If Analysis

## Overview
This document analyzes potential failure points in the preprocessing pipeline and provides mitigation strategies. Each scenario includes likelihood, impact, detection method, and resolution.

---

## SCENARIO MATRIX

| # | Scenario | Likelihood | Impact | Detection Time | Mitigation Priority |
|---|----------|------------|--------|----------------|---------------------|
| 1 | Input file missing/corrupted | Medium | High | Immediate | ⚠️ **CRITICAL** |
| 2 | GCS permission denied (read) | Low | High | <1 min | ⚠️ **CRITICAL** |
| 3 | GCS permission denied (write) | Low | High | 15-20 min | ⚠️ **CRITICAL** |
| 4 | Tokenizer download failure | Medium | High | 2-3 min | ⚠️ **HIGH** |
| 5 | Invalid JSON in dataset | Medium | Medium | Variable | ✅ **HANDLED** |
| 6 | Network timeout during processing | Low | Medium | Variable | ⚠️ **HIGH** |
| 7 | Disk space exhaustion | Very Low | High | Variable | ⚠️ **MEDIUM** |
| 8 | All examples filtered out | Low | High | 20 min | ⚠️ **HIGH** |
| 9 | Variance calculation underflow | Very Low | Low | 20 min | ✅ **HANDLED** |
| 10 | Unicode/encoding errors | Medium | Low | Variable | ✅ **HANDLED** |
| 11 | Service account token expiry | Very Low | High | Variable | ⚠️ **MEDIUM** |
| 12 | Quota exhaustion | Low | High | Variable | ⚠️ **HIGH** |
| 13 | Region availability issues | Very Low | High | Immediate | ⚠️ **CRITICAL** |
| 14 | Container startup timeout | Low | High | 5-10 min | ⚠️ **MEDIUM** |
| 15 | JSON serialization errors | Low | Medium | 20 min | ⚠️ **MEDIUM** |

---

## DETAILED SCENARIO ANALYSIS

### SCENARIO 1: Input File Missing or Corrupted
**Likelihood:** Medium (Human error, accidental deletion)  
**Impact:** High (Pipeline fails immediately)

**What Happens:**
```python
blob = bucket.blob(input_blob_name)  # Succeeds
with blob.open("r", encoding='utf-8') as f_in:  # ❌ FAILS HERE
```

**Error Message:**
```
google.api_core.exceptions.NotFound: 404 GET https://storage.googleapis.com/...
Blob not found: gs://newllm369-478400-llm-data/raw_data/openhermes-2.5-dataset.jsonl
```

**Detection:** Immediate (within 1 minute)

**Current Code Vulnerability:**
```python
# NO validation before processing!
blob = bucket.blob(input_blob_name)
with blob.open("r", encoding='utf-8') as f_in:  # Direct open
```

**Fix Required:**
```python
# Add validation
blob = bucket.blob(input_blob_name)
if not blob.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_data_path}\n"
        f"Please verify the file exists in GCS."
    )

# Check file size
blob.reload()
if blob.size == 0:
    raise ValueError(f"Input file is empty: {input_data_path}")

if blob.size < 1024:  # Less than 1KB is suspicious
    logger.warning(f"Input file is very small ({blob.size} bytes)")

logger.info(f"Input file size: {blob.size / (1024**3):.2f} GB")
```

**Mitigation Strategy:**
- Add pre-flight validation
- Check file existence and size
- Log file metadata before processing

---

### SCENARIO 2: GCS Permission Denied (Read)
**Likelihood:** Low (IAM already configured)  
**Impact:** High (Pipeline fails immediately)

**What Happens:**
```python
bucket = client.bucket(bucket_name)
blob = bucket.blob(input_blob_name)
with blob.open("r", encoding='utf-8') as f_in:  # ❌ PERMISSION DENIED
```

**Error Message:**
```
google.api_core.exceptions.Forbidden: 403 GET https://storage.googleapis.com/...
Permission denied: storage.objects.get
```

**Detection:** Within 1-2 minutes

**Current Code Vulnerability:**
- No permission pre-check
- Fails only when attempting to read

**Fix Required:**
```python
# Add permission check
try:
    bucket = client.bucket(bucket_name)
    bucket.reload()  # Verify bucket access
    
    blob = bucket.blob(input_blob_name)
    blob.reload()  # Verify blob access
    
    logger.info(f"✓ Permission check passed for {input_data_path}")
except Exception as e:
    raise PermissionError(
        f"GCS access denied. Service account needs storage.objects.get permission.\n"
        f"Error: {str(e)}"
    )
```

---

### SCENARIO 3: GCS Permission Denied (Write)
**Likelihood:** Low (IAM configured for write)  
**Impact:** High (Pipeline fails after 15-20 min of processing!)

**What Happens:**
```python
# Processing goes fine for 15-20 minutes...
train_blob = output_bucket.blob(f"{output_base_path}/train_data.jsonl")
with train_blob.open("w", encoding='utf-8') as f_train:  # ❌ FAILS HERE
```

**Error Message:**
```
google.api_core.exceptions.Forbidden: 403 POST https://storage.googleapis.com/...
Permission denied: storage.objects.create
```

**Detection:** 15-20 minutes (after processing starts)

**Current Code Vulnerability:**
- **CRITICAL**: Writes are attempted AFTER processing starts
- No pre-flight write permission check
- 20 minutes of compute wasted before failure

**Fix Required:**
```python
# VALIDATE WRITE PERMISSIONS BEFORE PROCESSING
output_bucket = client.bucket(output_bucket_name)

# Create a test file to verify write permissions
test_blob = output_bucket.blob(f"{output_base_path}/.write_test")
try:
    with test_blob.open("w") as f:
        f.write("permission_test")
    test_blob.delete()
    logger.info(f"✓ Write permission verified for {output_data_path}")
except Exception as e:
    raise PermissionError(
        f"GCS write access denied. Service account needs storage.objects.create.\n"
        f"Error: {str(e)}"
    )
```

**Priority:** ⚠️ **CRITICAL** - This is a HIGH RISK issue!

---

### SCENARIO 4: Tokenizer Download Failure
**Likelihood:** Medium (Network issues, HuggingFace down, rate limits)  
**Impact:** High (Pipeline fails early)

**What Happens:**
```python
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # ❌ FAILS
```

**Error Messages:**
```
# Network timeout
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='huggingface.co', port=443)

# Rate limit
requests.exceptions.HTTPError: 429 Too Many Requests

# Model not found
OSError: microsoft/DialoGPT-mediumm does not appear to be a valid model identifier
```

**Detection:** 2-5 minutes (during package installation + download)

**Current Code Vulnerability:**
```python
# No retry logic or error handling
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
```

**Fix Required:**
```python
import time
from requests.exceptions import RequestException

max_retries = 3
for attempt in range(max_retries):
    try:
        logger.info(f"Loading tokenizer: {tokenizer_name} (attempt {attempt + 1}/{max_retries})")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir="/tmp/tokenizers",  # Explicit cache
            local_files_only=False
        )
        logger.info(f"✓ Tokenizer loaded successfully")
        break
    except (RequestException, OSError) as e:
        if attempt == max_retries - 1:
            raise RuntimeError(
                f"Failed to load tokenizer after {max_retries} attempts.\n"
                f"Tokenizer: {tokenizer_name}\n"
                f"Error: {str(e)}\n"
                f"Possible causes:\n"
                f"  - Network connectivity issues\n"
                f"  - HuggingFace hub is down\n"
                f"  - Invalid tokenizer name\n"
                f"  - Rate limiting"
            )
        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in 10s...")
        time.sleep(10)
```

---

### SCENARIO 5: Invalid JSON in Dataset Lines
**Likelihood:** Medium (Data quality issues)  
**Impact:** Medium (Some examples skipped)

**What Happens:**
```python
data = json.loads(line.strip())  # ❌ JSONDecodeError
```

**Error Message:**
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Detection:** During processing (variable timing)

**Current Code Status:** ✅ **HANDLED**
```python
try:
    data = json.loads(line.strip())
    # ... processing
except Exception as e:
    logger.warning(f"Error processing line {line_num}: {str(e)[:100]}")
    skipped_examples += 1
    continue
```

**However, Risk Still Exists:**
- If ALL or MOST lines are invalid JSON, pipeline "succeeds" but produces no output
- No alerting if skip rate is too high

**Enhancement Required:**
```python
# After processing loop
skip_rate = (skipped_examples / total_lines * 100) if total_lines > 0 else 0

if skip_rate > 50:
    raise ValueError(
        f"CRITICAL: {skip_rate:.1f}% of examples were skipped!\n"
        f"Total lines: {total_lines:,}\n"
        f"Skipped: {skipped_examples:,}\n"
        f"Processed: {processed_count:,}\n"
        f"This indicates severe data quality issues."
    )
elif skip_rate > 20:
    logger.warning(
        f"HIGH skip rate detected: {skip_rate:.1f}% ({skipped_examples:,}/{total_lines:,})"
    )
```

---

### SCENARIO 6: Network Timeout During Processing
**Likelihood:** Low (GCS streaming is reliable)  
**Impact:** Medium (Pipeline restart needed)

**What Happens:**
```python
for line in f_in:  # ❌ Network timeout during streaming read
```

**Error Message:**
```
socket.timeout: timed out
OR
google.api_core.exceptions.ServiceUnavailable: 503 Service Unavailable
```

**Detection:** Variable (could happen at any point)

**Current Code Vulnerability:**
- No timeout configuration
- No retry on transient failures
- Loses all progress

**Fix Required:**
```python
from google.api_core import retry
from google.api_core.exceptions import ServiceUnavailable

# Configure retry policy
retry_policy = retry.Retry(
    initial=1.0,
    maximum=60.0,
    multiplier=2.0,
    predicate=retry.if_exception_type(
        ServiceUnavailable,
        ConnectionError,
    ),
)

# Apply to blob operations
blob = bucket.blob(input_blob_name)

# Use retry decorator for read operations
with blob.open("r", encoding='utf-8') as f_in:
    # GCS client already has built-in retries, but we can enhance
    pass
```

**Alternative: Checkpoint Strategy**
```python
# Save progress periodically
checkpoint_frequency = 100000  # Every 100k lines

if line_num > 0 and line_num % checkpoint_frequency == 0:
    checkpoint_data = {
        'last_line': line_num,
        'processed_count': processed_count,
        'train_count': train_count,
        'val_count': val_count,
    }
    checkpoint_blob = output_bucket.blob(f"{output_base_path}/.checkpoint.json")
    with checkpoint_blob.open("w") as f:
        json.dump(checkpoint_data, f)
```

---

### SCENARIO 7: Disk Space Exhaustion
**Likelihood:** Very Low (GCS streaming doesn't use local disk)  
**Impact:** High (Container crashes)

**What Happens:**
```python
# Unlikely since we stream directly to GCS, but could happen with:
# 1. /tmp fills up with tokenizer cache
# 2. Logging fills up disk
# 3. Python internal buffers
```

**Error Message:**
```
OSError: [Errno 28] No space left on device
```

**Detection:** Variable

**Current Code:** Mostly safe (streaming architecture)

**Enhancement:**
```python
import shutil

# Check disk space at start
total, used, free = shutil.disk_usage("/")
free_gb = free // (1024**3)

if free_gb < 5:  # Less than 5GB free
    logger.warning(f"Low disk space: {free_gb}GB free")
    
logger.info(f"Disk space: {free_gb}GB free / {total // (1024**3)}GB total")
```

---

### SCENARIO 8: All Examples Filtered Out
**Likelihood:** Low (Data quality or overly strict filtering)  
**Impact:** High (No training data produced)

**What Happens:**
```python
# All examples skipped due to:
# 1. No 'conversations' field
# 2. Empty conversations
# 3. Token length < 10
# 4. JSON parsing errors

# Pipeline "succeeds" but:
processed_count = 0
train_count = 0
val_count = 0
```

**Error:** No error raised! Pipeline reports success.

**Detection:** 20 minutes (only when checking outputs)

**Current Code Vulnerability:**
```python
# At the end, we calculate stats but don't validate
stats = {
    'processed_examples': processed_count,  # Could be 0!
    'train_examples': train_count,  # Could be 0!
    'val_examples': val_count,  # Could be 0!
}
```

**Fix Required:**
```python
# After processing, BEFORE saving stats
if processed_count == 0:
    raise ValueError(
        f"CRITICAL: Zero examples processed!\n"
        f"Total lines: {total_lines:,}\n"
        f"Skipped: {skipped_examples:,}\n"
        f"Possible causes:\n"
        f"  - All lines have invalid JSON\n"
        f"  - All conversations are empty\n"
        f"  - All tokenized sequences < 10 tokens\n"
        f"  - Wrong input file format"
    )

if processed_count < 100:
    logger.warning(
        f"Very few examples processed: {processed_count}\n"
        f"Expected: ~1M examples"
    )

if train_count == 0 or val_count == 0:
    raise ValueError(
        f"CRITICAL: Train or val split is empty!\n"
        f"Train: {train_count}, Val: {val_count}"
    )
```

**Priority:** ⚠️ **HIGH** - Silent failure scenario!

---

### SCENARIO 9: Variance Calculation Underflow
**Likelihood:** Very Low (would need extremely uniform data)  
**Impact:** Low (Minor stat error)

**What Happens:**
```python
variance = (sum_sq_lengths / processed_count - avg_length ** 2)
std_length = variance ** 0.5  # ❌ Negative variance causes error
```

**Error Message:**
```
ValueError: math domain error (square root of negative number)
```

**Detection:** 20 minutes (at stats calculation)

**Current Code Vulnerability:**
```python
variance = (sum_sq_lengths / processed_count - avg_length ** 2)
std_length = variance ** 0.5  # No check for negative variance
```

**Fix Required:**
```python
variance = (sum_sq_lengths / processed_count - avg_length ** 2) if processed_count > 0 else 0

# Handle floating point precision issues
if variance < 0 and variance > -1e-10:  # Tiny negative due to float precision
    variance = 0.0
    logger.warning("Variance adjusted from negative (float precision) to 0.0")
elif variance < 0:
    logger.error(f"Invalid negative variance: {variance}")
    raise ValueError(f"Calculated negative variance: {variance}")

std_length = variance ** 0.5 if variance > 0 else 0
```

---

### SCENARIO 10: Unicode/Encoding Errors
**Likelihood:** Medium (Dataset has international characters, emojis)  
**Impact:** Low (Individual examples skipped)

**What Happens:**
```python
chat_text += f"<|user|>{content}<|end|>\n"  # ❌ Unicode error
json.dumps(example, ensure_ascii=False)  # ❌ Serialization error
```

**Error Message:**
```
UnicodeEncodeError: 'ascii' codec can't encode character '\U0001f600'
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff
```

**Detection:** Variable (during processing)

**Current Code Status:** ✅ **MOSTLY HANDLED**
```python
# Already using UTF-8
with blob.open("r", encoding='utf-8') as f_in:
with train_blob.open("w", encoding='utf-8') as f_train:
json.dumps(example, ensure_ascii=False)  # Preserves unicode
```

**However, edge cases:**
```python
# Enhancement: Handle malformed UTF-8
with blob.open("r", encoding='utf-8', errors='replace') as f_in:
    # 'replace' will substitute � for bad bytes
```

---

### SCENARIO 11: Service Account Token Expiry
**Likelihood:** Very Low (Tokens auto-refresh)  
**Impact:** High (Mid-execution failure)

**What Happens:**
```python
# After 1 hour of processing, token expires
with train_blob.open("w", encoding='utf-8') as f_train:  # ❌ Unauthorized
```

**Error Message:**
```
google.auth.exceptions.RefreshError: The credentials do not support refreshing
```

**Detection:** 60+ minutes (during long runs)

**Current Code:** Google Cloud client libraries auto-refresh tokens

**Mitigation:** Already handled by google-cloud-storage library, but add monitoring:
```python
# Optional: Log credential status periodically
if line_num % 100000 == 0:
    logger.info(f"Processed {line_num:,} lines, credentials valid")
```

---

### SCENARIO 12: Quota Exhaustion
**Likelihood:** Low (Quotas are generous)  
**Impact:** High (Pipeline blocked)

**What Happens:**
```python
# Hit quota limits:
# - GCS operations/sec
# - GCS bandwidth
# - Vertex AI job concurrency
```

**Error Message:**
```
google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
OR
google.api_core.exceptions.TooManyRequests: 429 Too many requests
```

**Detection:** Variable

**Fix Required:**
```python
from google.api_core.exceptions import ResourceExhausted
import time

try:
    # GCS operations
    with blob.open("r") as f:
        pass
except ResourceExhausted as e:
    logger.error(f"Quota exceeded: {str(e)}")
    logger.info("Waiting 60s before retry...")
    time.sleep(60)
    # Retry logic
```

---

### SCENARIO 13: Region Availability Issues
**Likelihood:** Very Low (GCP regions are highly available)  
**Impact:** High (Cannot start pipeline)

**What Happens:**
```python
job = aiplatform.PipelineJob(...)
job.run()  # ❌ Region unavailable
```

**Error Message:**
```
google.api_core.exceptions.ServiceUnavailable: 503 
Region us-central1 is temporarily unavailable
```

**Detection:** Immediate (job submission fails)

**Fix:** No code fix - operational issue. Use multi-region deployment.

---

### SCENARIO 14: Container Startup Timeout
**Likelihood:** Low (Python 3.11 image is cached)  
**Impact:** High (5-10 min wasted)

**What Happens:**
```python
# Vertex AI tries to:
# 1. Pull python:3.11 image
# 2. Install packages
# 3. Start container
# Times out at any step
```

**Error Message:**
```
The replica workerpool0-0 failed to start: container startup timeout
```

**Detection:** 5-10 minutes

**Current Code:** Using standard base image (good)

**Mitigation:** Already optimal (using `python:3.11` not custom image)

---

### SCENARIO 15: JSON Serialization Errors
**Likelihood:** Low (Data types are standard)  
**Impact:** Medium (Example skipped or pipeline fails)

**What Happens:**
```python
example = {
    'input_ids': tokens['input_ids'],  # Could be non-serializable type
}
json.dumps(example, ensure_ascii=False)  # ❌ TypeError
```

**Error Message:**
```
TypeError: Object of type int64 is not JSON serializable
TypeError: Object of type ndarray is not JSON serializable
```

**Detection:** During processing

**Current Code Vulnerability:**
```python
example = {
    'input_ids': tokens['input_ids'],  # Could be numpy array or torch tensor
    'attention_mask': tokens['attention_mask'],
    'labels': tokens['input_ids'].copy(),  # .copy() is good, but type?
}
```

**Fix Required:**
```python
# Convert to native Python types
example = {
    'input_ids': [int(x) for x in tokens['input_ids']],  # Convert to list of ints
    'attention_mask': [int(x) for x in tokens['attention_mask']],
    'labels': [int(x) for x in tokens['input_ids']],
    'text': str(chat_text[:500] + "..." if len(chat_text) > 500 else chat_text),
    'source': str(data.get('source', 'unknown')),
    'category': str(data.get('category', 'unknown'))
}
```

---

## RECOMMENDED FIXES - PRIORITY ORDER

### 🔴 CRITICAL (Apply Immediately)

1. **Pre-flight Write Permission Check** (Scenario 3)
   - Prevents 20 min of wasted processing
   - Add before processing starts

2. **Input File Validation** (Scenario 1)
   - Check file exists and has valid size
   - Add at component start

3. **Zero Examples Validation** (Scenario 8)
   - Prevent silent failures
   - Add after processing loop

### 🟡 HIGH (Apply Soon)

4. **Tokenizer Download Retry** (Scenario 4)
   - Add 3-attempt retry with backoff
   - Handles transient network issues

5. **Skip Rate Monitoring** (Scenario 5)
   - Alert if >20% examples skipped
   - Raise error if >50% skipped

6. **JSON Serialization Safety** (Scenario 15)
   - Convert numpy/torch types to Python natives
   - Prevents mid-processing failures

### 🟢 MEDIUM (Nice to Have)

7. **Variance Underflow Protection** (Scenario 9)
   - Handle floating point precision issues
   - Prevent math domain errors

8. **Disk Space Monitoring** (Scenario 7)
   - Log available space at start
   - Warning if <5GB free

9. **Network Retry Policy** (Scenario 6)
   - Add explicit retry for GCS operations
   - Consider checkpoint strategy for very long runs

---

## IMPLEMENTATION PLAN

### Phase 1: Critical Fixes (Deploy ASAP)
```python
def preprocess_openhermes_data(...):
    # FIX 1: Input validation
    blob = bucket.blob(input_blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"Input not found: {input_data_path}")
    
    blob.reload()
    logger.info(f"Input file: {blob.size / (1024**3):.2f} GB")
    
    # FIX 2: Write permission check
    test_blob = output_bucket.blob(f"{output_base_path}/.write_test")
    try:
        with test_blob.open("w") as f:
            f.write("test")
        test_blob.delete()
    except Exception as e:
        raise PermissionError(f"Write access denied: {str(e)}")
    
    # ... processing ...
    
    # FIX 3: Zero examples check
    if processed_count == 0:
        raise ValueError("CRITICAL: Zero examples processed!")
```

### Phase 2: High Priority (Next iteration)
- Tokenizer retry logic
- Skip rate monitoring
- JSON type safety

### Phase 3: Medium Priority (Future enhancement)
- Checkpointing for restart capability
- Advanced monitoring and alerting
- Multi-region failover

---

## TESTING CHECKLIST

Before next run, verify:
- [ ] Input file exists in GCS
- [ ] Service account has read permission (storage.objects.get)
- [ ] Service account has write permission (storage.objects.create)
- [ ] Output bucket/path is writable
- [ ] Tokenizer name is correct: "microsoft/DialoGPT-medium"
- [ ] Memory limit 32GB is configured
- [ ] CPU limit 8 cores is configured
- [ ] Network connectivity to HuggingFace hub
- [ ] GCS quotas are not exhausted
- [ ] No other jobs consuming quota

---

## MONITORING DURING EXECUTION

Watch for these indicators:

✅ **Healthy Signs:**
- "Loading tokenizer..." appears within 3 minutes
- "Processing line 10,000..." logs appear regularly (every ~10 seconds)
- Processing rate: ~500-1000 lines/second
- Skip rate stays <5%

⚠️ **Warning Signs:**
- No log output for >2 minutes
- Skip rate >20%
- Processing slows down significantly
- Memory usage climbing steadily

🔴 **Critical Issues:**
- Pipeline stuck at tokenizer loading (>5 min)
- "Permission denied" errors
- "Quota exceeded" errors
- No "Processing line" logs after 5 minutes

---

## CONCLUSION

**Total Scenarios Analyzed:** 15  
**Already Handled:** 3 (Scenarios 5, 10, 11)  
**Require Fixes:** 12  
**Critical Priority:** 3  
**High Priority:** 3  
**Medium Priority:** 6  

**Recommended Action:** Apply Critical fixes before next run. This will catch 80% of potential failures early and save significant debugging time.
