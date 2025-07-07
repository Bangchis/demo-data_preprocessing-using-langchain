from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import streamlit as st
import re

class AgentManager:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            temperature=0
        )
        
    def create_agent(self, df: pd.DataFrame):
        """Create pandas dataframe agent"""
        
        agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type="openai-tools"
        )
        
        return agent
    
    def extract_code_from_response(self, response: str) -> list:
        """Extract Python code blocks from agent response"""
        
        # Pattern to match code blocks
        code_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
        ]
        
        extracted_codes = []
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                code = match.strip()
                if 'df' in code and len(code) > 10:
                    extracted_codes.append(code)
        
        return extracted_codes
    
    def execute_code_safely(self, code: str) -> tuple:
        """Execute pandas code and update session state"""
        
        try:
            # Get current state
            current_dfs = st.session_state.get("dfs", {})
            current_df = st.session_state.get("df", None)
            
            if current_df is None:
                return False, "No dataframe available"
            
            # Create execution environment
            exec_globals = {
                'pd': pd,
                'df': current_df.copy(),
                'dfs': current_dfs.copy(),
                'np': __import__('numpy'),
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Update session state with results
            if 'df' in exec_globals:
                updated_df = exec_globals['df']
                if isinstance(updated_df, pd.DataFrame):
                    st.session_state.df = updated_df
                    return True, f"✅ DataFrame updated! Shape: {updated_df.shape}"
                else:
                    return False, f"❌ Invalid DataFrame result"
            else:
                return False, "❌ Code didn't update 'df' variable"
                
        except Exception as e:
            return False, f"❌ Execution error: {str(e)}"
    
    def process_query(self, query: str, df: pd.DataFrame) -> str:
        """Process user query with actual code execution"""
        try:
            agent = self.create_agent(df)
            
            # ENHANCED PROMPT FOR CODE EXECUTION
            enhanced_query = f"""
            You are an EXPERT data preprocessing assistant.
            
            USER REQUEST: {query}
            
            CRITICAL: You must provide executable pandas code in ```python blocks.
            
            WORKFLOW:
            1. 🎯 UNDERSTAND: Analyze the request
            2. 🔧 PLAN: Design steps
            3. 💻 EXECUTE: Write executable code
            4. ✅ VERIFY: Ensure correctness
            
            CODE REQUIREMENTS:
            - Always work with 'df' variable
            - Write code in ```python blocks
            - Ensure df is updated: df = df.some_operation()
            - Use clear, executable pandas code
            
            RESPONSE FORMAT:
            ### 🎯 Analysis
            [Understanding of request]
            
            ### 🔧 Approach
            [Step-by-step plan]
            
            ### 💻 Code Implementation
            ```python
            # Executable pandas code here
            # Update df variable
            df = df.some_operation()
            ```
            
            ### 📊 Results
            [Expected results]
            
            ### 💡 Insights & Recommendations
            [Key insights and next steps]
            
            IMPORTANT: Code in ```python blocks will be executed!
            """
            
            # Get agent response
            response = agent.run(enhanced_query)
            
            # Extract and execute code
            extracted_codes = self.extract_code_from_response(response)
            
            execution_results = []
            code_executed = False
            
            for code in extracted_codes:
                success, result = self.execute_code_safely(code)
                execution_results.append(f"**Code Block:** `{code[:50]}...`\n**Result:** {result}")
                if success:
                    code_executed = True
            
            # Add execution status to response
            if code_executed:
                response += "\n\n🔄 **CODE EXECUTION STATUS:**\n" + "\n\n".join(execution_results)
                response += f"\n\n✅ **DataFrame successfully updated! New shape: {st.session_state.df.shape}**"
            else:
                response += "\n\n⚠️ **No executable code found or execution failed.**"
                if execution_results:
                    response += "\n\n**Execution attempts:**\n" + "\n\n".join(execution_results)
            
            return response
            
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"