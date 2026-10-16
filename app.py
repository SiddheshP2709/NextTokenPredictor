"""Standard Streamlit Root Entry Point.

Delegates execution to web/app.py.
"""

import os
import runpy

TARGET_APP = os.path.join(os.path.dirname(__file__), "web", "app.py")

if __name__ == "__main__":
    runpy.run_path(TARGET_APP, run_name="__main__")
