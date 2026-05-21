from .shell_tool import ShellTool

def ShellToolLC():
    try:
        from ..integrations.langchain_adapter import LangchainToolAdapter
        return LangchainToolAdapter.to_langchain_structured_tool(ShellTool())
    except ImportError as e:
        raise ImportError("LangChain core components are not installed. Install 'langchain-core' or use agent factory auto-adaptation.") from e

__all__ = ['ShellTool', 'ShellToolLC']