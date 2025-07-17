from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import pandas as pd
import streamlit as st
import re
import datetime
from typing import List, Dict, Any

# Import token management
from src.utils.token_manager import TokenManager, get_token_manager

# Import all tools
from src.tools.core import (
    CodeRunnerTool, UndoTool, RedoTool, ExecutionLogTool
)
from src.tools.basic import (
    QuickInfoTool, MissingReportTool, DuplicateCheckTool, ColumnSummaryTool,
    ValueCountsTool, CorrelationTool, OutlierTool, SchemaReportTool, FullInfoTool,
    StructuralErrorAnalysisTool, BasicStructuralCheckTool
)
from src.tools.web import (
    WebSearchTool, PandasHelpTool, DataScienceHelpTool, ErrorSolutionTool, SearchHistoryTool
)
from src.tools.backup_tools import (
    ManualBackupTool, ListBackupsTool, RestoreBackupTool, BackupStatsTool, 
    DeleteBackupTool, QuickBackupTool, CleanupBackupsTool
)


class AgentManager:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            temperature=0
        )
        self.tools = self._initialize_tools()
        self.agent = None
        self.max_context_messages = 10  # Giới hạn số lượng messages trong context
        self.token_manager = get_token_manager(model_name)  # Token management
        
    def _initialize_tools(self) -> List:
        """Initialize all available tools"""
        return [
            # Core execution tools
            CodeRunnerTool,
            UndoTool,
            RedoTool,
            ExecutionLogTool,
            
            # Data exploration tools
            QuickInfoTool,
            FullInfoTool,
            MissingReportTool,
            DuplicateCheckTool,
            ColumnSummaryTool,
            ValueCountsTool,
            CorrelationTool,
            OutlierTool,
            SchemaReportTool,
            StructuralErrorAnalysisTool,
            BasicStructuralCheckTool,
            
            # Web search tools
            WebSearchTool,
            PandasHelpTool,
            DataScienceHelpTool,
            ErrorSolutionTool,
            SearchHistoryTool,
            
            # Backup tools
            ManualBackupTool,
            ListBackupsTool,
            RestoreBackupTool,
            BackupStatsTool,
            DeleteBackupTool,
            QuickBackupTool,
            CleanupBackupsTool
        ]
    
    def create_agent(self):
        """Create ReAct agent with tools"""
        
        # ReAct prompt template in Vietnamese
        prompt_prefix = """
Bạn là DataProcessingAgent – một chuyên gia phân tích và xử lý dữ liệu có khả năng sử dụng các công cụ để khám phá, làm sạch và báo cáo dữ liệu một cách hệ thống và an toàn.

Bạn có quyền truy cập vào DataFrame chính (st.session_state.df) cùng các công cụ dưới đây để thực hiện nhiệm vụ.

🔧 CÔNG CỤ XỬ LÝ & THỰC THI:
• CodeRunner        – Thực thi mã pandas an toàn (TUYỆT ĐỐI KHÔNG ĐƯỢC DÙNG COMMENT #)
• Undo              – Hoàn tác thao tác gần nhất
• Redo              – Lặp lại thao tác đã hoàn tác
• ExecutionLog      – Truy vấn lịch sử các lần thực thi

📊 CÔNG CỤ PHÂN TÍCH & THỐNG KÊ:
• QuickInfo         – Tóm tắt info(), describe(), và phát hiện lỗi cấu trúc
• FullInfo          – Báo cáo toàn diện (schema, missing, duplicates, outliers, structural rows)
• MissingReport     – Phân tích dữ liệu thiếu theo cột
• DuplicateCheck    – Xác định dòng trùng lặp
• ColumnSummary     – Phân loại cột theo kiểu dữ liệu
• ValueCounts       – Thống kê top giá trị trong cột cụ thể
• CorrelationMatrix – Liệt kê cặp cột có tương quan cao
• OutlierCheck      – Đếm outlier theo IQR
• SchemaReport      – Báo cáo dtype & missing value
• StructuralErrorAnalysis – Phân tích lỗi cấu trúc nâng cao (thiếu ID, collapse cột, pattern không nhất quán)
• BasicStructuralCheck    – Kiểm tra lỗi cấu trúc cơ bản (để so sánh)

🌐 CÔNG CỤ HỖ TRỢ TRA CỨU:
• WebSearch         – Tìm thông tin trên Internet (Tavily real-time, 100 req/day free)
• PandasHelp        – Tìm cú pháp Pandas (tự động filter pandas.pydata.org, stackoverflow.com)
• DataScienceHelp   – Hỏi về kiến thức thống kê, tiền xử lý (kaggle.com, towardsdatascience.com)
• ErrorSolution     – Tìm giải pháp cho lỗi cụ thể (stackoverflow.com, github.com)
• SearchHistory     – Truy vấn lịch sử tìm kiếm

💾 CÔNG CỤ BACKUP & PHỤC HỒI:
• ManualBackup      – Tạo backup thủ công với tên và mô tả
• QuickBackup       – Tạo backup nhanh với tên timestamp
• ListBackups       – Liệt kê các backup có sẵn
• RestoreBackup     – Khôi phục từ backup theo ID
• BackupStats       – Thống kê backup (tổng số, dung lượng, lần gần nhất)
• DeleteBackup      – Xóa backup theo ID
• CleanupBackups    – Dọn dẹp backup tự động cũ

🎯 PHƯƠNG PHÁP LÀM VIỆC:
Bạn sẽ nhận được câu hỏi và phải trả lời bằng cách sử dụng các công cụ có sẵn. Hãy suy nghĩ từng bước và sử dụng công cụ phù hợp để thu thập thông tin.

Sau cùng, PHẢI trả lời bằng định dạng sau:

Final Answer: 📊 EXECUTIVE SUMMARY
[Tóm tắt 2–3 câu về tình trạng tổng quan]

🔍 CHI TIẾT FINDINGS

[Tên vấn đề] ([Priority Level]): [Mô tả chi tiết, có số liệu cụ thể]

[...] [...]

💡 RECOMMENDATIONS

Immediate: [...]

Short-term: [...]

Long-term: [...]

📋 NEXT STEPS

[...]

[...]

[...]

📈 IMPACT ASSESSMENT
[Phân tích tác động nếu vấn đề không được xử lý hoặc giải pháp được áp dụng]

⚠️ NGUYÊN TẮC NGHIÊM NGẶT VỀ CODE:
- **🚫 TUYỆT ĐỐI KHÔNG GỬI COMMENT (#)** trong đoạn mã - KHÔNG CÓ NGOẠI LỆ!
- **🚫 KHÔNG DÙNG MARKDOWN** (```python) trong Action Input của CodeRunner
- **🚫 KHÔNG GIẢI THÍCH** trong code, chỉ gửi code thuần
- **🚫 KHÔNG DÙNG st.session_state.df** - chỉ dùng **df** trong code
- **✅ CHỈ GỬI CODE PYTHON THUẦN** - một dòng lệnh duy nhất
- **✅ LUÔN DÙNG CodeRunner** cho mọi thao tác code

📋 NGUYÊN TẮC NGHIÊM NGẶT VỀ REPORT:
- Final Answer PHẢI bắt đầu bằng "Final Answer:" và đủ định dạng chi tiết – KHÔNG viết tắt, KHÔNG bỏ phần nào.
- Luôn trình bày rõ ràng reasoning và giải thích từng bước.
- Ngôn ngữ sử dụng: **Tiếng Việt**
- Sử dụng PRIORITY LEVEL: HIGH / MEDIUM / LOW cho từng Finding.
- Tất cả recommendation PHẢI actionable – có bước cụ thể.
- Luôn kiểm tra bằng QuickInfo hoặc SchemaReport trước khi xử lý sâu.
- Có thể khai thác thông tin quá khứ từ chat_context nếu cần thiết.

💾 NGUYÊN TẮC BACKUP QUAN TRỌNG:
- **TRƯỚC KHI THỰC HIỆN BẤT KỲ THAO TÁC RỦI RO NÀO**, hãy dùng **ManualBackup** hoặc **QuickBackup** để tạo backup
- Các thao tác RỦI RO bao gồm: drop, delete, transform, fillna, merge, split, apply
- Nếu có lỗi xảy ra, hãy dùng **RestoreBackup** để khôi phục về trạng thái trước
- Định kỳ dùng **BackupStats** để kiểm tra tình trạng backup
- Nếu cần xem các backup có sẵn, dùng **ListBackups**
- Luôn khuyến khích người dùng tạo backup trước khi thực hiện thao tác quan trọng

🔥 ĐỊNH DẠNG ĐÚNG cho CodeRunner:
Action: CodeRunner
Action Input: df.dropna().head()

🔥 ĐỊNH DẠNG ĐÚNG khác:
Action: CodeRunner
Action Input: df.info()

🔥 ĐỊNH DẠNG ĐÚNG khác:
Action: CodeRunner
Action Input: df.describe()

⚠️ LƯU Ý QUAN TRỌNG:
- CHỈ GỬI CODE PYTHON THUẦN
- KHÔNG MARKDOWN, KHÔNG COMMENT, KHÔNG GIẢI THÍCH
- CHỈ CÓ MỘT DÒNG CODE DUY NHẤT

Hãy sử dụng các công cụ có sẵn để phân tích dữ liệu một cách hệ thống và cung cấp kết quả chi tiết.
"""
        
        # Create ReAct agent
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=30,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        
        # The agent already has the correct ReAct template, just modify the system prompt
        # Get the original template and update it with our Vietnamese instructions
        original_template = agent.agent.llm_chain.prompt.template
        
        # Replace the default system message with our Vietnamese prompt
        # Keep the original LangChain template structure but update the instructions
        updated_template = original_template.replace(
            "Answer the following questions as best you can. You have access to the following tools:",
            f"{prompt_prefix}\n\nAnswer the following questions as best you can. You have access to the following tools:"
        )
        
        # Set the updated template
        agent.agent.llm_chain.prompt.template = updated_template
        
        return agent
    
    def process_query(self, query: str, df: pd.DataFrame = None) -> str:
        """Process user query with ReAct agent"""
        try:
            # Set current time for logging
            st.session_state.current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create agent if not exists
            if self.agent is None:
                self.agent = self.create_agent()
            
            # Validate and optimize query length
            query_info = self.token_manager.get_query_token_info(query)
            if query_info["is_over_limit"]:
                # Show warning to user
                warning_msg = f"⚠️ **Query quá dài**: {query_info['tokens']} tokens (giới hạn: {query_info['limit']})"
                st.warning(warning_msg)
                
                # Optimize query
                optimized_query = self.token_manager.smart_truncate_query(query)
                st.info(f"✂️ **Query đã được rút gọn**: {self.token_manager.count_tokens(optimized_query)} tokens")
                query = optimized_query
            
            # Lấy context từ lịch sử chat
            chat_context_messages = self.get_chat_context_messages()
            
            # System instructions
            system_instructions = self.get_system_instructions()
            
            # Optimize query and context for token budget
            optimized_query, optimized_context = self.token_manager.optimize_context_for_query(
                query, chat_context_messages, system_instructions
            )
            
            # Generate chat context string from optimized context
            chat_context = self.format_chat_context(optimized_context)
            
            # Enhanced query with context
            enhanced_query = f"""
{chat_context}🎯 YÊU CẦU CỦA NGƯỜI DÙNG: {optimized_query}

📊 THÔNG TIN NGỮ CẢNH:
- DataFrame hiện tại: {st.session_state.df.shape if st.session_state.df is not None else 'Chưa tải dữ liệu'}
- Thời gian: {st.session_state.current_time}

🔍 HƯỚNG DẪN:
1. Phân tích yêu cầu kỹ lưỡng (tham khảo lịch sử cuộc hội thoại nếu có)
2. Sử dụng công cụ phù hợp để khám phá/xử lý dữ liệu
3. Khi cần thực thi code, sử dụng CodeRunner
4. Khi cần tìm hiểu thêm, sử dụng WebSearch hoặc PandasHelp
5. Giải thích rõ ràng từng bước
6. **QUAN TRỌNG**: Final Answer phải chi tiết, có số liệu cụ thể và actionable
7. **PRIORITY LEVELS**: Sử dụng HIGH/MEDIUM/LOW cho mọi vấn đề được phát hiện
8. **QUANTITATIVE**: Luôn cung cấp số liệu cụ thể (%, số lượng, tỷ lệ) khi có thể
9. **CONTEXT-AWARE**: Tham chiếu đến các phân tích trước đó từ lịch sử chat
10. **ACTIONABLE**: Mọi recommendation phải có bước thực hiện cụ thể

Hãy bắt đầu với Thought để phân tích yêu cầu:
"""
            
            # Get token usage summary for debugging
            token_summary = self.token_manager.get_token_usage_summary(
                system_instructions, enhanced_query, optimized_context
            )
            
            # Show token usage if warnings exist
            if token_summary["warnings"]:
                with st.expander("🔍 Token Usage Details"):
                    for warning in token_summary["warnings"]:
                        st.warning(warning)
                    
                    st.json({
                        "Token Breakdown": token_summary["breakdown"],
                        "Usage": token_summary["usage"]
                    })
            
            # Get agent response
            response = self.agent.run(enhanced_query)
            
            # Clean up response format
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            # Try to get error solution
            error_msg = str(e)
            try:
                # Search for error solution
                error_solution = ErrorSolutionTool.func(error_msg)
                return f"❌ **Lỗi xử lý:** {error_msg}\n\n🔍 **Tìm kiếm giải pháp:**\n{error_solution}"
            except:
                return f"❌ Lỗi xử lý truy vấn: {error_msg}"
    
    def _clean_response(self, response: str) -> str:
        """Clean and format agent response"""
        # Remove excessive whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        
        # Format for better readability
        response = response.strip()
        
        return response
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.name for tool in self.tools]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get tool descriptions for UI display"""
        return {tool.name: tool.description for tool in self.tools}
    
    def get_chat_context(self) -> str:
        """Tạo context từ lịch sử chat để tích hợp vào prompt"""
        if "react_chat_history" not in st.session_state or not st.session_state.react_chat_history:
            return ""
        
        # Lấy số lượng messages gần đây nhất (giới hạn để tránh quá tải context)
        recent_messages = st.session_state.react_chat_history[-self.max_context_messages:]
        
        if not recent_messages:
            return ""
        
        context_parts = ["📚 **LỊCH SỬ CUỘC HỘI THOẠI GẦN ĐÂY:**"]
        
        for i, msg in enumerate(recent_messages):
            role = "👤 Người dùng" if msg["role"] == "user" else "🤖 Agent"
            # Cắt ngắn nội dung nếu quá dài
            content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
            context_parts.append(f"{role}: {content}")
        
        context_parts.append("---")
        return "\n".join(context_parts) + "\n\n"
    
    def get_chat_context_messages(self) -> List[Dict]:
        """Get chat context messages as list of dictionaries"""
        if "react_chat_history" not in st.session_state or not st.session_state.react_chat_history:
            return []
        
        return st.session_state.react_chat_history[-self.max_context_messages:]
    
    def get_system_instructions(self) -> str:
        """Get system instructions for token calculation"""
        return """
Bạn là DataProcessingAgent – một chuyên gia phân tích và xử lý dữ liệu có khả năng sử dụng các công cụ để khám phá, làm sạch và báo cáo dữ liệu một cách hệ thống và an toàn.

Bạn có quyền truy cập vào DataFrame chính (st.session_state.df) cùng các công cụ dưới đây để thực hiện nhiệm vụ.

🔧 CÔNG CỤ XỬ LÝ & THỰC THI:
• CodeRunner        – Thực thi mã pandas an toàn (TUYỆT ĐỐI KHÔNG ĐƯỢC DÙNG COMMENT #)
• Undo              – Hoàn tác thao tác gần nhất
• Redo              – Lặp lại thao tác đã hoàn tác
• ExecutionLog      – Truy vấn lịch sử các lần thực thi

📊 CÔNG CỤ PHÂN TÍCH & THỐNG KÊ:
• QuickInfo         – Tóm tắt info(), describe(), và phát hiện lỗi cấu trúc
• FullInfo          – Báo cáo toàn diện (schema, missing, duplicates, outliers, structural rows)
• MissingReport     – Phân tích dữ liệu thiếu theo cột
• DuplicateCheck    – Xác định dòng trùng lặp
• ColumnSummary     – Phân loại cột theo kiểu dữ liệu
• ValueCounts       – Thống kê top giá trị trong cột cụ thể
• CorrelationMatrix – Liệt kê cặp cột có tương quan cao
• OutlierCheck      – Đếm outlier theo IQR
• SchemaReport      – Báo cáo dtype & missing value
• StructuralErrorAnalysis – Phân tích lỗi cấu trúc nâng cao (thiếu ID, collapse cột, pattern không nhất quán)
• BasicStructuralCheck    – Kiểm tra lỗi cấu trúc cơ bản (để so sánh)

🌐 CÔNG CỤ HỖ TRỢ TRA CỨU:
• WebSearch         – Tìm thông tin trên Internet (Tavily real-time, 100 req/day free)
• PandasHelp        – Tìm cú pháp Pandas (tự động filter pandas.pydata.org, stackoverflow.com)
• DataScienceHelp   – Hỏi về kiến thức thống kê, tiền xử lý (kaggle.com, towardsdatascience.com)
• ErrorSolution     – Tìm giải pháp cho lỗi cụ thể (stackoverflow.com, github.com)
• SearchHistory     – Truy vấn lịch sử tìm kiếm

🎯 PHƯƠNG PHÁP LÀM VIỆC:
Bạn sẽ nhận được câu hỏi và phải trả lời bằng cách sử dụng các công cụ có sẵn. Hãy suy nghĩ từng bước và sử dụng công cụ phù hợp để thu thập thông tin.

Sau cùng, PHẢI trả lời bằng định dạng sau:

Final Answer: 📊 EXECUTIVE SUMMARY
[Tóm tắt 2–3 câu về tình trạng tổng quan]

🔍 CHI TIẾT FINDINGS

[Tên vấn đề] ([Priority Level]): [Mô tả chi tiết, có số liệu cụ thể]

[...] [...]

💡 RECOMMENDATIONS

Immediate: [...]

Short-term: [...]

Long-term: [...]

📋 NEXT STEPS

[...]

[...]

[...]

📈 IMPACT ASSESSMENT
[Phân tích tác động nếu vấn đề không được xử lý hoặc giải pháp được áp dụng]

⚠️ NGUYÊN TẮC NGHIÊM NGẶT VỀ CODE:
- **🚫 TUYỆT ĐỐI KHÔNG GỬI COMMENT (#)** trong đoạn mã - KHÔNG CÓ NGOẠI LỆ!
- **🚫 KHÔNG DÙNG MARKDOWN** (```python) trong Action Input của CodeRunner
- **🚫 KHÔNG GIẢI THÍCH** trong code, chỉ gửi code thuần
- **🚫 KHÔNG DÙNG st.session_state.df** - chỉ dùng **df** trong code
- **✅ CHỈ GỬI CODE PYTHON THUẦN** - một dòng lệnh duy nhất
- **✅ LUÔN DÙNG CodeRunner** cho mọi thao tác code

📋 NGUYÊN TẮC NGHIÊM NGẶT VỀ REPORT:
- Final Answer PHẢI bắt đầu bằng "Final Answer:" và đủ định dạng chi tiết – KHÔNG viết tắt, KHÔNG bỏ phần nào.
- Luôn trình bày rõ ràng reasoning và giải thích từng bước.
- Ngôn ngữ sử dụng: **Tiếng Việt**
- Sử dụng PRIORITY LEVEL: HIGH / MEDIUM / LOW cho từng Finding.
- Tất cả recommendation PHẢI actionable – có bước cụ thể.
- Luôn kiểm tra bằng QuickInfo hoặc SchemaReport trước khi xử lý sâu.
- Có thể khai thác thông tin quá khứ từ chat_context nếu cần thiết.

💾 NGUYÊN TẮC BACKUP QUAN TRỌNG:
- **TRƯỚC KHI THỰC HIỆN BẤT KỲ THAO TÁC RỦI RO NÀO**, hãy dùng **ManualBackup** hoặc **QuickBackup** để tạo backup
- Các thao tác RỦI RO bao gồm: drop, delete, transform, fillna, merge, split, apply
- Nếu có lỗi xảy ra, hãy dùng **RestoreBackup** để khôi phục về trạng thái trước
- Định kỳ dùng **BackupStats** để kiểm tra tình trạng backup
- Nếu cần xem các backup có sẵn, dùng **ListBackups**
- Luôn khuyến khích người dùng tạo backup trước khi thực hiện thao tác quan trọng

🔥 ĐỊNH DẠNG ĐÚNG cho CodeRunner:
Action: CodeRunner
Action Input: df.dropna().head()

🔥 ĐỊNH DẠNG ĐÚNG khác:
Action: CodeRunner
Action Input: df.info()

🔥 ĐỊNH DẠNG ĐÚNG khác:
Action: CodeRunner
Action Input: df.describe()

⚠️ LƯU Ý QUAN TRỌNG:
- CHỈ GỬI CODE PYTHON THUẦN
- KHÔNG MARKDOWN, KHÔNG COMMENT, KHÔNG GIẢI THÍCH
- CHỈ CÓ MỘT DÒNG CODE DUY NHẤT

Hãy sử dụng các công cụ có sẵn để phân tích dữ liệu một cách hệ thống và cung cấp kết quả chi tiết.
"""
    
    def format_chat_context(self, context_messages: List[Dict]) -> str:
        """Format context messages into string for prompt"""
        if not context_messages:
            return ""
        
        context_parts = ["📚 **LỊCH SỬ CUỘC HỘI THOẠI GẦN ĐÂY:**"]
        
        for msg in context_messages:
            role = "👤 Người dùng" if msg["role"] == "user" else "🤖 Agent"
            # Use token manager to truncate content if needed
            content = msg["content"]
            if self.token_manager.count_tokens(content) > 200:
                content = self.token_manager.truncate_query(content, preserve_ratio=0.8)
            context_parts.append(f"{role}: {content}")
        
        context_parts.append("---")
        return "\n".join(context_parts) + "\n\n"
    
    def set_max_context_messages(self, max_messages: int):
        """Cấu hình giới hạn số lượng messages trong context"""
        self.max_context_messages = max_messages
    
    def clear_chat_history(self):
        """Xóa lịch sử chat"""
        if "react_chat_history" in st.session_state:
            st.session_state.react_chat_history = []
            return True
        return False
    
    def get_chat_stats(self) -> Dict[str, Any]:
        """Lấy thống kê về lịch sử chat"""
        if "react_chat_history" not in st.session_state:
            return {"total_messages": 0, "context_messages": 0, "memory_active": False}
        
        total_messages = len(st.session_state.react_chat_history)
        context_messages = min(total_messages, self.max_context_messages)
        
        return {
            "total_messages": total_messages,
            "context_messages": context_messages,
            "memory_active": context_messages > 0,
            "max_context_messages": self.max_context_messages
        }
    
    def reset_agent(self):
        """Reset agent (useful when changing models)"""
        self.agent = None