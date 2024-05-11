import os
from pydantic import BaseModel
from typing import Optional, Any
import ast
from llama_index.llms.ollama import Ollama
from llama_index.core import  PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts.prompt import context, code_parser_template
from .code_reader import code_reader
from dotenv import load_dotenv
import os
import ast
from config.toolsconfig import get_tools
from docAgent.docllama import DocumentQueryEngine

class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

class CodeGenerator:
    def __init__(self, document_engine: DocumentQueryEngine, output_parser: Any, query_pipeline: Any):
        self.document_engine = document_engine
        self.output_parser = output_parser
        self.query_pipeline = query_pipeline
        self.output_directory = "output"
        self.code_llm = Ollama(model="codellama")
        os.makedirs(self.output_directory, exist_ok=True)
        self.agent = ReActAgent.from_tools(self.document_engine.tools, llm=self.code_llm , verbose=True, context=context)


    def generate_code(self, prompt: str) -> Optional[CodeOutput]:
        print("this is toos:", self.document_engine)
        retries = 0
        while retries < 3:
            try:
                result = self.agent.query(prompt)
                next_result = self.query_pipeline.run(response=result)
                cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
                return CodeOutput(**cleaned_json)
                break
            except Exception as e:
                retries += 1
                print(f"Error occurred, retry #{retries}:", e)

        
        print("Unable to process request after 3 retries.")
        return None

    def save_code(self, code_output: CodeOutput):
        try:
            filepath = os.path.join(self.output_directory, code_output.filename)
            with open(filepath, "w") as file:
                file.write(code_output.code)
            print("Saved file:", code_output.filename)
        except Exception as e:
            print("Error saving file:", e)

    def run(self):
        while (prompt := input("Enter a prompt (q to quit): ")) != "q":
            code_output = self.generate_code(prompt)
            if code_output:
                print("Code generated")
                print(code_output.code)
                print("\n\nDescription:", code_output.description)
                self.save_code(code_output)

# Setup and use the CodeGenerator
if __name__ == "__main__":
    # Setup dependencies
    document_query_engine = DocumentQueryEngine() 
    llm = document_query_engine.llm
    code_output_parser = PydanticOutputParser(CodeOutput)
    json_prompt_str = code_output_parser.format(code_parser_template)
    json_prompt_tmpl = PromptTemplate(json_prompt_str)
    output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

    code_generator = CodeGenerator(document_engine=document_query_engine, output_parser=code_output_parser, query_pipeline=output_pipeline)
    code_generator.run()
