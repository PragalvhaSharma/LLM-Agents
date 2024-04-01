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


from dotenv import load_dotenv
load_dotenv()

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

#Loading model
print(db.get_usable_table_names()) # Printing names to see if we loaded the database right 

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613')

sqlAgent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True #Handling error by itself if it runs into a problerm with input
)

search = SerpAPIWrapper()

@tool
def SQLAgentTool(query: str) -> str:
    """Executes SQL queries or questions about the company itself. Uses the following tables to answer questions:{db.get_usable_table_names()} """
    return sqlAgent.invoke(query)

@tool
def searchTool(query: str) -> str:
    """Performs web searches for current events or information outside the LLM scope E.g Date."""
    return search.run(query)

tools = [SQLAgentTool, searchTool]
toolNames = ["SQLAgentTool", "searchTool"]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your name is Hermes. Your primary role is to assist users by providing accurate information about their business or anyother topic they might require assistance with.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


llm_with_tools = llm.bind_tools(tools)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
output = list(agent_executor.stream({"input": "what did i ask you before"}))