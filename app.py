# Import necessary libraries from Flask and other custom modules
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import os

os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_KEY")
# Initialize the OpenAI ChatGPT model
model = ChatOpenAI(openai_api_key=os.getenv("CHATGPT_KEY"))

# Read the content of a file named 'promptior.txt'
with open('promptior.txt', 'r') as file:
    text = file.read()
    
# Create a RecursiveCharacterTextSplitter object to split the text into smaller documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
# Generate documents from the text using the text_splitter
docs = text_splitter.create_documents([text])

# Create a vector store using the OpenAI embeddings model
vector_store = FAISS.from_texts([text], embedding=OpenAIEmbeddings())

# Create a vector store retriever
vector_store_retriever = vector_store.as_retriever()

#Define a system template providing context for answering questions
SYSTEM_TEMPLATE = """Use the following pieces of context to answer the question at the end.
The special characters change to HTML Formatter.
If you don't know the answer, all write only ":(". 
----------------
{context}"""

# Create a list of messages including a system message and a human message
messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]

# Generate a prompt template using the messages
prompt = ChatPromptTemplate.from_messages(messages)

# Assemble a runnable chain combining the vector store context, question, and the ChatGPT model
chain = (
    {"context": vector_store_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

#Setting up Flask Application: A Flask web application is initialized, defining routes for the chat interface.
app = Flask(__name__)

#Defining Routes:
# /: Renders the main chat HTML template.
# /get: Handles GET and POST requests, retrieving user input and returning the chat response.
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    input = request.form["msg"]
    return get_chat_response(input)

#Chat Response Function: The get_chat_response function takes user input, passes it through the LangChain processing chain, and returns the generated response.
def get_chat_response(text):
    # Invoke the chain to get a response
    response = chain.invoke(text)
    
    # Check for a sad emoticon in the response
    if ':(' in response:
        # If sad emoticon is present, invoke the model directly
        response = model.invoke(text)
        return response.content
    else:
        return response

if __name__ == '__main__':
    # Run the Flask application
    app.run()
