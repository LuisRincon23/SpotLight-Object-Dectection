#!/usr/bin/env python3
"""
SpotLight Web Application Launcher
Run this script to start the web interface
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from webapp import app, init_camera, init_model

if __name__ == '__main__':
    print("🔦 SpotLight - Real-time Object Detection")
    print("=" * 40)
    print("Initializing camera and model...")
    
    try:
        init_camera()
        init_model()
        print("✅ Initialization complete!")
        print("\n📱 Starting web server...")
        print("🌐 Open http://localhost:8080 in your browser")
        print("\nPress Ctrl+C to stop the server")
        
        app.run(debug=False, threaded=True, port=8080, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down SpotLight...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check that your camera is connected and you have the required dependencies installed.")