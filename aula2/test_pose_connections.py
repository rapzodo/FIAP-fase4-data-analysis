#!/usr/bin/env python3
"""
Test to check if MediaPipe provides POSE_CONNECTIONS constant.
This verifies whether we need the fallback definition or not.
"""

import sys

print("=" * 60)
print("Testing MediaPipe POSE_CONNECTIONS availability")
print("=" * 60)

# Test 1: Try mediapipe.python.solutions.pose
print("\n1. Testing: from mediapipe.python.solutions.pose import POSE_CONNECTIONS")
try:
    from mediapipe.python.solutions.pose import POSE_CONNECTIONS
    print("   ✓ SUCCESS!")
    print(f"   - Type: {type(POSE_CONNECTIONS)}")
    print(f"   - Count: {len(POSE_CONNECTIONS)} connections")
    print(f"   - Sample: {list(POSE_CONNECTIONS)[:3]}")
    sys.exit(0)
except ImportError as e:
    print(f"   ✗ ImportError: {e}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Try mediapipe.solutions.pose (legacy)
print("\n2. Testing: from mediapipe.solutions.pose import POSE_CONNECTIONS")
try:
    from mediapipe.solutions.pose import POSE_CONNECTIONS
    print("   ✓ SUCCESS!")
    print(f"   - Type: {type(POSE_CONNECTIONS)}")
    print(f"   - Count: {len(POSE_CONNECTIONS)} connections")
    sys.exit(0)
except ImportError as e:
    print(f"   ✗ ImportError: {e}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Check mediapipe.python module exists
print("\n3. Testing: import mediapipe.python")
try:
    import mediapipe.python
    print("   ✓ mediapipe.python module exists")
    print(f"   - Contents: {dir(mediapipe.python)}")
except ImportError as e:
    print(f"   ✗ ImportError: {e}")

print("\n" + "=" * 60)
print("CONCLUSION: POSE_CONNECTIONS is NOT available in MediaPipe 0.10.31")
print("The fallback definition in the code is NECESSARY.")
print("=" * 60)

