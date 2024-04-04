import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from dotenv import load_dotenv
load_dotenv()

# Retrieve API keys from environment variables
myOpenAIkey = os.environ["OPENAI_API_KEY"]
print(myOpenAIkey)
googleSearchKey = os.environ["SERPAPI_API_KEY"]

print(myOpenAIkey)
#Database INFO
userName = "postgres"
password = "984138o35o"
host = "localhost"
port = "5432"
mydatabase = "mynewdatabase"
pg_uri = f"postgresql+psycopg2://{userName}:{password}@{host}:{port}/{mydatabase}"
db = SQLDatabase.from_uri(pg_uri)

# Print available table names to verify database connection
print(db.get_usable_table_names()) 

#Initialize llm
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613', api_key=myOpenAIkey)

# Create an SQL agent with predefined toolkits and settings
sqlAgent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True
)

# Initialize a wrapper for SerpAPI
search = SerpAPIWrapper()

# Define a tool function for the SQLAgent

@tool()
def SQLAgentTool(query: str) -> str:
    """Executes SQL queries or questions about the company itself using the SQLAgent. If you identify the question from the user to be a SQL query, pass the user input directly to the agent. Pass the relevant table name to getTableINFOforSQL to get information about the table prior to running the SQL query .
    E.g If the user says: How many work orders are finished? --- The user is asking how many work orders are set to true for the column finished.

       """
    try:
        # Attempt to invoke the SQL agent with the provided query
        result = sqlAgent.run(query)
        logging.info(f"Query executed successfully: {query}")
        return result
    except Exception as e:
        # Log any exceptions that occur during the query execution
        logging.error(f"Error executing query: {query}, Error: {str(e)}")
        # Optionally, return a user-friendly message or handle the error appropriately
        return "There was an error processing your SQL query. Please try again."

# Define a tool function for performing web searches
@tool
def searchTool(query: str) -> str:
    """Performs web searches for current events or information outside the LLM's scope of information."""
    return search.run(query)

##Placed incase for agent
@tool()
def getTableINFOforSQL(tableName: str) -> str:
    """Called for every SQL query. Provides information about a given table from the database. It is used whenever the user asks question regarding the sql database"""
    return db.get_table_info([tableName])

# List of available tools and their names
tools = [SQLAgentTool, searchTool, getTableINFOforSQL]
toolNames = ["SQLAgentTool", "searchTool", "getTableINFOforSQL"]

print(db.get_usable_table_names())
print(db.get_table_info(["workorders"]))

# Initialize memory for conversation history and add initial message for the user
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory.chat_memory.add_ai_message("Hello! My name is Hermes. How may I help you today?")
memory.chat_memory.add_user_message(f"Hello! These are the tables in my sql database. Make sure to understand the relationship between them {db.get_usable_table_names()} ")


MEMORY_KEY = "chat_history"

# Prepare prompt template for conversation
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Your name is Hermes. Your primary role is to assist users by providing accurate information about their business or any other topic they might require assistance with. IF the query is reagarding SQL, pass the user question directly onto the SQLAgentTool and retrieve the output. "),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Bind the tools to the language model
llm_with_tools = llm.bind_tools(tools)
chat_history = []

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

input1 = "How many tables are there in the database "
result = agent_executor.invoke({"input": input1, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)

print(agent_executor.invoke({"input": "how many work orders were finished ", "chat_history": chat_history}))
print(agent_executor.invoke({"input": "how many work orders are there ", "chat_history": chat_history}))
print(agent_executor.invoke({"input": "What is the price of Ethereum.", "chat_history": chat_history}))
