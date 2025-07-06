from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import streamlit as st

class AgentManager:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            temperature=0
        )
        
    def create_agent(self, df: pd.DataFrame):
        """Create pandas dataframe agent without Docker sandbox (for now)"""
        
        agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,  # Will add Docker sandbox later
            agent_type="openai-tools"
        )
        
        return agent
    
    def process_query(self, query: str, df: pd.DataFrame) -> str:
        """Process user query with enhanced instruction prompt"""
        try:
            agent = self.create_agent(df)
            
            # ENHANCED INSTRUCTION PROMPT v2.0
            enhanced_query = f"""
            You are an EXPERT data preprocessing assistant with deep pandas knowledge and excellent communication skills.
            
            USER REQUEST: {query}
            
            WORKFLOW (Follow this exactly):
            1. 🎯 UNDERSTAND: Break down what the user wants to achieve
            2. 🔧 PLAN: Design the logical steps needed
            3. 💻 EXECUTE: Write clean, efficient pandas code
            4. ✅ VERIFY: Check results and provide insights
            5. 📋 SUMMARIZE: Give actionable conclusions
            
            CODE STANDARDS:
            - Always work with the 'df' variable 
            - Write readable, well-commented code
            - Handle edge cases (missing values, data types, empty results)
            - Use vectorized operations for performance
            - Follow pandas best practices
            - Update df when transformations are needed
            
            RESPONSE FORMAT (Use this structure):
            ### 🎯 Analysis
            [What you understand the user wants]
            
            ### 🔧 Approach
            [Your step-by-step plan]
            
            ### 💻 Code Implementation
            [The pandas code with detailed comments]
            
            ### 📊 Results
            [Display the results clearly with key metrics]
            
            ### 💡 Insights & Recommendations
            [Key findings, patterns, and actionable next steps]
            
            COMMON OPERATIONS EXAMPLES:
            - Data Exploration: df.info(), df.describe(), df.head(), df.shape
            - Cleaning: df.dropna(), df.fillna(), df.drop_duplicates()
            - Filtering: df[df['col'] > value], df.query('condition')
            - Transforming: df['new'] = df['col'].apply(func)
            - Aggregating: df.groupby('col').agg({{'metric': 'mean'}}).reset_index()
            - Sorting: df.sort_values('col', ascending=False)
            - Statistics: df['col'].mean(), df['col'].std(), df['col'].quantile()
            
            QUALITY REQUIREMENTS:
            ✓ Does the code solve the user's exact request?
            ✓ Are there any data quality issues to highlight?
            ✓ Can you suggest relevant follow-up analyses?
            ✓ Is the explanation clear for both technical and non-technical users?
            ✓ Are the results properly formatted and easy to understand?
            
            IMPORTANT: Always provide insights and business value, not just code execution!
            """
            
            response = agent.run(enhanced_query)
            return response
            
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"