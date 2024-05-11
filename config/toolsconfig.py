from llama_index.core.tools import QueryEngineTool, ToolMetadata
from codeAgent.code_reader import code_reader

def get_tools(query_engine):
    return [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                 name="resume_info",
                    description="this gives documentation about a personal resume. Good for summarizing resume",
                ),
        ),
         code_reader,
    ]

  