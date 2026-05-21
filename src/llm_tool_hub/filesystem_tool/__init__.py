from .read_file_tool import ReadFileTool
from .create_file_tool import CreateFileTool
from .modify_file_tool import ModifyFileTool
from .replace_content_tool import ReplaceContentTool

def ReadFileToolLC():
    try:
        from ..integrations.langchain_adapter import LangchainToolAdapter
        return LangchainToolAdapter.to_langchain_structured_tool(ReadFileTool())
    except ImportError as e:
        raise ImportError("LangChain core components are not installed. Install 'langchain-core' or use agent factory auto-adaptation.") from e

def CreateFileToolLC():
    try:
        from ..integrations.langchain_adapter import LangchainToolAdapter
        return LangchainToolAdapter.to_langchain_structured_tool(CreateFileTool())
    except ImportError as e:
        raise ImportError("LangChain core components are not installed. Install 'langchain-core' or use agent factory auto-adaptation.") from e

def ModifyFileToolLC():
    try:
        from ..integrations.langchain_adapter import LangchainToolAdapter
        return LangchainToolAdapter.to_langchain_structured_tool(ModifyFileTool())
    except ImportError as e:
        raise ImportError("LangChain core components are not installed. Install 'langchain-core' or use agent factory auto-adaptation.") from e

def ReplaceContentToolLC():
    try:
        from ..integrations.langchain_adapter import LangchainToolAdapter
        return LangchainToolAdapter.to_langchain_structured_tool(ReplaceContentTool())
    except ImportError as e:
        raise ImportError("LangChain core components are not installed. Install 'langchain-core' or use agent factory auto-adaptation.") from e

__all__ = [
    'ReadFileTool',
    'CreateFileTool',
    'ModifyFileTool',
    'ReplaceContentTool',
    'ReadFileToolLC',
    'CreateFileToolLC',
    'ModifyFileToolLC',
    'ReplaceContentToolLC'
]