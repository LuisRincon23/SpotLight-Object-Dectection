#!/usr/bin/env python3
"""
SpotLight CLI Launcher
Run this script for command-line real-time detection
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from realtime_all_items import main

if __name__ == '__main__':
    print("🔦 SpotLight CLI - Real-time Object Detection")
    print("=" * 45)
    main()