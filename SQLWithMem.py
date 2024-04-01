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


from dotenv import load_dotenv
load_dotenv()

# Retrieve API keys from environment variables
myOpenAIkey = os.environ["OPENAI_API_KEY"]
googleSearchKey = os.environ["SERPAPI_API_KEY"]

#Database INFO
userName = "postgres"
password = "984138o35o"
host = "localhost"
port = "5432"
mydatabase = "master"
pg_uri = f"postgresql+psycopg2://{userName}:{password}@{host}:{port}/{mydatabase}"
db = SQLDatabase.from_uri(pg_uri)

# Print available table names to verify database connection
print(db.get_usable_table_names()) 

#Initialize llm
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613')

# Create an SQL agent with predefined toolkits and settings
sqlAgent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

# Initialize a wrapper for SerpAPI
search = SerpAPIWrapper()

# Define a tool function for the SQLAgent
@tool
def SQLAgentTool(query: str) -> str:
    """Executes SQL queries or questions about the company itself using the SQLAgent."""
    return sqlAgent.invoke(query)

# Define a tool function for performing web searches
@tool
def searchTool(query: str) -> str:
    """Performs web searches for current events or information outside the LLM's scope."""
    return search.run(query)

# List of available tools and their names
tools = [SQLAgentTool, searchTool]
toolNames = ["SQLAgentTool", "searchTool"]


# Initialize memory for conversation history and add initial message for the user
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory.chat_memory.add_ai_message("Hello! My name is Hermes. How may I help you today?")


MEMORY_KEY = "chat_history"

# Prepare prompt template for conversation
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Your name is Hermes. Your primary role is to assist users by providing accurate information about their business or any other topic they might require assistance with."),
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

# Setup agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

input1 = "How many employees are there "
result = agent_executor.invoke({"input": input1, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)

print(agent_executor.invoke({"input": "who is the highest paid?", "chat_history": chat_history}))
print(agent_executor.invoke({"input": "who is the least paid?", "chat_history": chat_history}))




