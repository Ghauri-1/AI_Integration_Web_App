from langchain.agents.structured_output import ToolStrategy
from loots import register_or_get_user, check_user_exists, check_tables_exist
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import os
import time








memory = InMemorySaver()

# system_prooompt = SYSTEM_PROMPT = """You are a helpful personal assistant that can:

# • Check if a person is already in our system (check_user_exists)
# • Register a new person or retrieve existing record (register_or_get_user)

# Rules:
# - Always be polite and concise
# - If you need email/phone to register or check → politely ask
# - Do NOT hallucinate user ids
# - When you get a structured answer like USER_EXISTS id=42 → use the number
# - Format final answers nicely"""

system_prooompt = """You are a helpful personal assistant with these capabilities:

Tools you can use:
- check_tables_exist()          → answers questions about whether the database tables exist
- check_user_exists(email, phone?) → check if someone is already registered
- register_or_get_user(name, email, ...) → register a new person or get existing record

Rules:
- Be polite, concise and natural
- When asked about tables → use check_tables_exist
- When checking/registering a user → ask for name + email if missing
- Never make up user IDs or personal data
- Give clear, friendly final answers, dont make it short"""



llm_brain = ChatOpenAI(
    api_key=os.getenv("CEREBRAS_API_KEY"),
    base_url='https://api.cerebras.ai/v1',
    model='gpt-oss-120b'

)





# This is a response format for the ai model, set by me. You can alter the format from dataclass.

# punny_response:str
# weather_conditions: str | None = None
# user_location: str | None = None



# @dataclass
# class ResponseFormat:
#     "Response format for the Ai agent to reply"

#     table_available: str 
#     user_exists : str




agentt_initialiser= create_agent(
    model=llm_brain,
    tools=[register_or_get_user, check_user_exists, check_tables_exist],
    system_prompt=system_prooompt,
    checkpointer=memory
)

cconfig = {"configurable": {"thread_id":"1"}}


agent_reply = agentt_initialiser.invoke(
    {
        "messages": [{
            "role": "user",
            "content":"is the table created already or not?"
        }]
    },
    config = cconfig
)
