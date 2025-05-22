from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent,create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from tools.home_price_tool import predict_home_price
from tools.anomaly_detection_tool import predict_anomalies
from tools.home_type_tool import predict_home_type
from tools.investment_tool import predict_investment
from tools.dataframe_tool import view_dataframe
import uuid
import asyncio

BASE_URL = "http://ollama_service:11434"

# Ollama modelini streaming modunda başlat
model = ChatOllama(
    model="qwen2.5:latest",
    base_url=BASE_URL,
    streaming=True,
)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a real estate assistant.

You have 5 tools to help users with properties located in Istanbul:

---

🔧 **Available Tools:**

1. **`predict_home_price`** – Predicts the price of a home.  
   Required information:  
   - `GrossSquareMeters` (float): Gross square meters  
   - `ItemStatus` (string): Listing status ('Boş', 'Eşyalı')  
   - `room` (float): Number of rooms  
   - `hall` (float): Number of halls  
   - `district` (string): District name  

   ▶️ If any of these details are missing, explicitly ask the user for them.  
   ✅ Once all info is provided, use this tool.  
   🔄 Response format:  
   `Estimated price: 3,250,000 TRY`  
   followed by a brief explanation of how the price was estimated.

2. **`predict_anomalies`** – Detects anomalies in the property data.  
   🔄 Response format:  
   - `✅ Data is normal.`  
   - `⚠️ Data appears to be anomalous.`

3. **`predict_home_type`** – Identifies the type category of the home.  
   Use only if the user asks about similar home types.  
   🔄 Response classes:  
   - Suitable Apartments for Middle-Income Families  
   - Luxury and Spacious Residences for High-Income Group  
   - Mid-Segment, Spacious and Economical Residences  
   - Multi-Room Ultra-Luxury Living Spaces

4. **`predict_investment`** – Evaluates if the home is a good investment.  
   Use only if the user requests investment advice.  
   🔄 Response format:  
   - `✅ Suitable for investment.`  
   - `⚠️ Risky. Caution advised when investing.`

5. **`view_dataframe`** – Used to visualize property data with pandas. 
   Use it when user wants to list something. Summarize content.
   Maximum of 3 properties can be listed.  
   🔄 Response: A dictionary containing general information about the properties.

---

📌 **General Rules:**

- If the user's question requires tools, **first collect missing information**, then use the tools **in the order above**.  
- If the user is making casual conversation, greetings, or general talk, **do not use any tools**. Just continue the chat naturally.  
- When discussing data (e.g., gross square meters, price, room count), **use terms consistent with the user's language**.  
- Only deal with properties in Istanbul. If asked about other cities, clarify you don't have info on those.  
- **Always convert district names to lowercase and replace Turkish characters with their English equivalents.**  
- **If the user writes room numbers like "3+1", always rephrase it as "3 rooms and 1 hall".**  
- **Never explain code, functions, or technical details to the user.**  
- Use past knowledge.
- All districts in Istanbul are valid. Do not limit yourself to the example districts.(district:adalar, arnavutkoy, atasehir, avcilar, bagcilar, bahcelievler, bakirkoy, basaksehir, bayrampasa, besiktas, beykoz, beylikduzu, beyoğlu, buyukcekmece, catalca, cekmekoy, esenler, esenyurt, eyupsultan, fatih, gaziosmanpasa, gungoren, kadikoy, kagithane, kartal, katalca, kucukcekmece, maltepe, pendik, sancaktepe, sultanbeyli, sultangazi, sancaktepe, sarıyer, silivri, sultanbeyli, sultangazi, sisli, tuzla, umraniye, uskudar, zeytinburnu)
"""
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])



tools = [predict_home_price,predict_anomalies,predict_home_type,predict_investment,view_dataframe]
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

session_id = str(uuid.uuid4())
memory = ChatMessageHistory(session_id=session_id)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    
)

async def ask(question: str):
    config = RunnableConfig(configurable={"session_id": session_id})
    content = ""
    async for chunk in agent_with_chat_history.astream({"input": question}, config=config):
        output = chunk.get("output")
        if output is not None:
            content += output
            yield content