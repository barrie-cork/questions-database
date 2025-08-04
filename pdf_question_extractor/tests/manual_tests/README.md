# Manual Test Scripts

This directory contains manual test scripts that are not part of the automated test suite.

## Scripts

### test_ocr_direct.py
Direct test of Mistral OCR API to verify metadata availability and API response format.

**Usage:**
```bash
python tests/manual_tests/test_ocr_direct.py
```

### test_ocr_metadata.py
Test script to inspect detailed OCR metadata extraction from Mistral API.

**Usage:**
```bash
python tests/manual_tests/test_ocr_metadata.py
```

### test_setup.py
Verify all project dependencies are installed correctly.

**Usage:**
```bash
python tests/manual_tests/test_setup.py
```

## Note
These scripts are meant for manual testing and debugging. They are not run as part of the standard test suite (`pytest`).