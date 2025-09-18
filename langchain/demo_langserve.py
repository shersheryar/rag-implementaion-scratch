from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
import uvicorn
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))  # should print your key

# create prompt templete

system_prompt = "translate the following into {language}"

prompt_template  = ChatPromptTemplate([
    ("system" , system_prompt),
    ("user" , "{text}")
])

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)

parser = StrOutputParser()

chain = prompt_template | model | parser

app  = FastAPI(
    title="my llm api",
    description="my first llm api",
    version="1.0"
)

add_routes(
    app,
    chain,
    path="/chain"
    
)

if __name__ =="__main__":
    uvicorn.run(app, host="localhost", port=8000)
    