#!/usr/bin/env python3
"""
Check data_cot directory JSON files for format compliance.

This script validates that all train/val/test(_xxx)/dev/eval.json files in data_cot
directory meet the required format specifications.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_json_files(data_cot_dir: str) -> List[str]:
    """Find all relevant JSON files in data_cot directory."""
    json_files = []
    data_cot_path = Path(data_cot_dir)

    if not data_cot_path.exists():
        print(f"Error: Directory {data_cot_dir} does not exist")
        return json_files

    # Patterns to match: train.json, val.json, test.json, test_xxx.json, dev.json, eval.json
    patterns = [
        "**/train.json",
        "**/val.json",
        "**/test.json",
        "**/test_*.json",
        "**/dev.json",
        "**/eval.json",
    ]

    for pattern in patterns:
        for file_path in data_cot_path.glob(pattern):
            json_files.append(str(file_path))

    return sorted(json_files)


def check_redacted_reasoning_tag(value: str, has_semantic: bool = False) -> Tuple[bool, str]:
    """Check if value contains <think></think> tags and validates content.
    
    Args:
        value: The value string from conversations[1]['value']
        has_semantic: Whether conversations[0]['value'] contains "semantic"
    """
    if "<think>" not in value or "</think>" not in value:
        return False, "Missing <think> or </think> tags"

    # Extract content between tags
    match = re.search(r'<think>(.*?)</think>', value, re.DOTALL)
    if not match:
        return False, "Could not extract content between tags"

    reasoning_content = match.group(1)

    # Check 1: Must contain transcription
    transcription_pattern = r'The transcription of this utterance is:\s*"[^"]*"'
    if not re.search(transcription_pattern, reasoning_content):
        return False, "Missing 'The transcription of this utterance is: \"...\"' in reasoning content"

    # Check what comes after </think>
    after_tag = value[value.find("</think>") + len("</think>"):]

    # Check 3: Must have two newlines after </think>
    if not after_tag.startswith("\n\n"):
        return False, "Must have exactly two newlines (\\n\\n) after </think>"

    # Check final statement
    after_newlines = after_tag[2:]  # Skip the two newlines

    # Special check for semantic: if has_semantic, must be spoof with influence
    if has_semantic:
        expected_prefix = "The utterance is a spoof. The spoofing operation may result in the following influence:"
        if not after_newlines.startswith(expected_prefix):
            return False, "If conversations[0]['value'] contains 'semantic', answer must start with 'The utterance is a spoof. The spoofing operation may result in the following influence:'"
    else:
        # General check for non-semantic cases
        if not (
            after_newlines.startswith("The utterance is a spoof.") or
            after_newlines.startswith("The utterance is real.")
        ):
            return False, "After \\n\\n, must start with 'The utterance is a spoof.' or 'The utterance is real.'"

    # Check 2: If it's a spoof, check for required content
    if "The utterance is a spoof." in after_newlines:
        # Must have spoof method (with or without quotes)
        # Pattern: "This indicates the spoof method is" followed by method name (with or without quotes) ending with period
        spoof_method_pattern = r'This indicates the spoof method is\s*[^.\n]+\.'
        if not re.search(spoof_method_pattern, reasoning_content):
            return False, "For spoof utterances, must contain 'This indicates the spoof method is ...' in reasoning content"

        # Must have one of: entire utterance, fake region, or fake regions
        has_entire = "The entire utterance is manipulated." in reasoning_content
        # Match "The fake region is: xxx-xxx seconds."
        has_fake_region = bool(re.search(r'The fake region is:\s*[0-9.]+-[0-9.]+ seconds\.', reasoning_content))
        # Match "The fake regions are: xxx-xxx seconds, yyy-yyy seconds."
        # Also supports "The fake regions are: (xxx-xxx seconds, yyy-yyy seconds)."
        # Pattern: optional opening paren, one or more "xxx-xxx seconds" separated by commas, optional closing paren
        fake_regions_pattern = r'The fake regions are:\s*(?:\([0-9.]+-[0-9.]+ seconds(?:,\s*[0-9.]+-[0-9.]+ seconds)*\)|[0-9.]+-[0-9.]+ seconds(?:,\s*[0-9.]+-[0-9.]+ seconds)*)\.'
        has_fake_regions = bool(re.search(fake_regions_pattern, reasoning_content))

        if not (has_entire or has_fake_region or has_fake_regions):
            return False, "For spoof utterances, must contain one of: 'The entire utterance is manipulated.', 'The fake region is: xxx-xxx seconds.', or 'The fake regions are: xxx-xxx seconds, ...'"

    return True, "OK"


def check_file(file_path: str) -> Tuple[bool, List[Tuple[int, str]]]:
    """Check a single JSON file for compliance."""
    errors = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [(0, f"JSON decode error: {e}")]
    except Exception as e:
        return False, [(0, f"Error reading file: {e}")]

    if not isinstance(data, list):
        return False, [(0, "Data is not a list")]

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append((idx, "Item is not a dictionary"))
            continue

        if 'conversations' not in item:
            errors.append((idx, "Missing 'conversations' field"))
            continue

        conversations = item['conversations']
        if not isinstance(conversations, list) or len(conversations) < 2:
            errors.append((
                idx,
                f"conversations must be a list with at least 2 items, got {len(conversations) if isinstance(conversations, list) else type(conversations)}"
            ))
            continue

        if 'value' not in conversations[1]:
            errors.append((idx, "Missing 'value' field in conversations[1]"))
            continue

        # Check if conversations[0]['value'] contains "semantic"
        has_semantic = False
        if 'value' in conversations[0]:
            has_semantic = "semantic" in conversations[0]['value'].lower()

        value = conversations[1]['value']
        is_valid, error_msg = check_redacted_reasoning_tag(value, has_semantic=has_semantic)

        if not is_valid:
            errors.append((idx, error_msg))

    return len(errors) == 0, errors


def main():
    """Main function."""
    data_cot_dir = "data_cot"

    if len(sys.argv) > 1:
        data_cot_dir = sys.argv[1]

    print(f"Checking JSON files in: {data_cot_dir}\n")

    json_files = find_json_files(data_cot_dir)

    if not json_files:
        print(f"No matching JSON files found in {data_cot_dir}")
        return 1

    print(f"Found {len(json_files)} JSON files to check:\n")
    for f in json_files:
        print(f"  - {f}")
    print()

    all_passed = True
    total_files = 0
    total_items = 0
    total_errors = 0

    for file_path in json_files:
        total_files += 1
        is_valid, errors = check_file(file_path)

        # Count items in file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_items += len(data) if isinstance(data, list) else 0
        except:
            pass

        if is_valid:
            print(f"✓ {file_path} - OK")
        else:
            all_passed = False
            total_errors += len(errors)
            print(f"✗ {file_path} - {len(errors)} error(s):")
            for idx, error_msg in errors[:10]:  # Show first 10 errors per file
                print(f"    Item {idx}: {error_msg}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more error(s)")

    print()
    print("=" * 60)
    print(f"Summary:")
    print(f"  Files checked: {total_files}")
    print(f"  Total items: {total_items}")
    print(f"  Total errors: {total_errors}")
    print(f"  Status: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
