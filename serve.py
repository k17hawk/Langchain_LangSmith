from fastapi import FastAPI
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes  # creates api
import uvicorn

# Load environment variables
load_dotenv()

# Constants
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")
groq_api_key = os.getenv('GROQ_API')

# Model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt template
generic_template = "Translate the following language {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", generic_template), ("user", "{text}")]
)

# Parsing the output
parser = StrOutputParser()

# Create a chain
chain = prompt_template | model | parser

# App definition
app = FastAPI(
    title="Langchain Server",
    version="0.0",
    description="A simple demo project for testing LCEL using lang serve"
)

# Adding the routes
add_routes(
    app,
    chain,
    path="/chain"
) 

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
