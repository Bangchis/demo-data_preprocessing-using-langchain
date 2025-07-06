import docker
import tempfile
import os
import time
from typing import Dict, Optional
import pandas as pd
from utils import save_dataframes_to_pickle, load_dataframes_from_pickle
import streamlit as st

class DockerSandbox:
    def __init__(self):
        self.client = docker.from_env()
        self.temp_dir = os.path.join(os.getcwd(), "temp_data")
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def execute_pandas_code(self, code: str, dfs: Dict, df: Optional[pd.DataFrame]) -> tuple:
        """Execute pandas code in Docker container safely"""
        
        # Prepare input data
        input_path = os.path.join(self.temp_dir, "input.pkl")
        output_path = os.path.join(self.temp_dir, "output.pkl")
        
        # Save current dataframes
        save_dataframes_to_pickle(dfs, df, input_path)
        
        # Prepare Python script to run in container
        script_content = f'''
import pandas as pd
import pickle
import sys

try:
    # Load data
    with open("/app/data/input.pkl", "rb") as f:
        data = pickle.load(f)
    
    dfs = data["dfs"]
    df = data["df"]
    
    # Make dataframes available in global scope
    globals().update(dfs)
    if df is not None:
        globals()["df"] = df
    
    # Execute user code
{code}
    
    # Save results
    result_data = {{
        "dfs": dfs,
        "df": df if "df" in globals() else None,
        "success": True,
        "output": "Code executed successfully"
    }}
    
    with open("/app/data/output.pkl", "wb") as f:
        pickle.dump(result_data, f)
        
except Exception as e:
    result_data = {{
        "dfs": dfs,
        "df": df,
        "success": False,
        "output": f"Error: {{str(e)}}"
    }}
    
    with open("/app/data/output.pkl", "wb") as f:
        pickle.dump(result_data, f)
'''
        
        # Write script to temp file
        script_path = os.path.join(self.temp_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(script_content)
        
        try:
            # Run container
            container = self.client.containers.run(
                "python:3.11-slim",
                command=["python", "/app/script.py"],
                volumes={
                    self.temp_dir: {"bind": "/app/data", "mode": "rw"},
                    script_path: {"bind": "/app/script.py", "mode": "ro"}
                },
                mem_limit="256m",
                network_disabled=True,
                detach=True,
                remove=True
            )
            
            # Wait for completion (with timeout)
            result = container.wait(timeout=30)
            
            if result["StatusCode"] == 0:
                # Load results
                with open(output_path, "rb") as f:
                    result_data = pickle.load(f)
                
                return (
                    result_data["dfs"],
                    result_data["df"],
                    result_data["success"],
                    result_data["output"]
                )
            else:
                return dfs, df, False, "Container execution failed"
                
        except docker.errors.ContainerError as e:
            return dfs, df, False, f"Container error: {str(e)}"
        except Exception as e:
            return dfs, df, False, f"Execution error: {str(e)}"
        finally:
            # Cleanup
            for file in [input_path, output_path, script_path]:
                if os.path.exists(file):
                    os.remove(file)