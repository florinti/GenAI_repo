# ---------- Importing Required Libraries ----------
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import requests
import logging
import uvicorn
import math
from datetime import datetime
import json
import os


# ---------- Logging Setup ----------
# Basic configuration for logging to console.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log to a file named agent_llm.log with timestamped messages.
file_handler = logging.FileHandler("agent_llm.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Clears the conversation log file each time the app restarts.
log_filepath = "agent_llm_conversations.json"
with open(log_filepath, "w") as f:
    f.write("")  # Start with empty file


# ---------- FastAPI App Initialization ----------
# Creates an instance of the FastAPI application.
app = FastAPI(title="LangChain App")


# ---------- Request Model ----------
# Defines the structure of incoming POST requests (expects a JSON with "question").
class QuestionRequest(BaseModel):
    question: str


# ---------- Custom Callback Handler ----------
# This class logs each significant step taken by the agent (prompting, tool use, etc.)
class FileLoggingCallbackHandler(BaseCallbackHandler):
    def __init__(self, filepath: str = log_filepath):
        self.filepath = filepath
        self.last_tool = None
        self.agent_started = False
        self.agent_final_output = None

    # Helper function to write logs to file in JSON format.
    def _log(self, label: str, payload: dict):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": label,
            "payload": payload
        }
        with open(self.filepath, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # Logs when a chain (agent execution) starts.
    def on_chain_start(self, serialized, inputs, **kwargs):
        if not self.agent_started:
            self.agent_started = True
            self._log("chain_start", {
                "serialized": serialized,
                "inputs": inputs
            })

    # Logs when the agent sends a prompt to the LLM.
    def on_llm_start(self, serialized, prompts, **kwargs):
        for i, prompt in enumerate(prompts):
            self._log("agent_to_llm_prompt_natural", {
                "message": "Agent sent the following prompt to LLM:",
                "prompt_index": i,
                "full_prompt_text": prompt
            })

    # Logs the LLM's response to the agent.
    def on_llm_end(self, response, **kwargs):
        try:
            full_text = response.generations[0][0].text if response.generations and response.generations[0] else "<no text>"
        except Exception:
            full_text = "<error extracting LLM output>"
        self._log("llm_to_agent_response_natural", {
            "message": "LLM responded with the following text:",
            "full_response_text": full_text
        })

    # Logs when the agent starts using a tool.
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.last_tool = serialized.get("name", "UnknownTool")
        self._log("agent_to_tool_query", {
            "message": f"Agent queried tool '{self.last_tool}' with input: '{input_str}'",
            "tool_name": self.last_tool,
            "input": input_str,
            "serialized": serialized
        })

    # Logs the tool's response back to the agent.
    def on_tool_end(self, output: str, **kwargs):
        self._log("tool_to_agent_response", {
            "tool_name": self.last_tool,
            "output": output
        })
        self.last_tool = None

    # Logs the final output produced by the agent.
    def on_chain_end(self, outputs, **kwargs):
        self.agent_final_output = outputs.get("output") if isinstance(outputs, dict) else str(outputs)
        self._log("agent_final_output", {
            "output": self.agent_final_output,
            "full_outputs": outputs
        })


# Instantiate the callback logger once so it can be reused globally.
callback_handler = FileLoggingCallbackHandler()


# ---------- LLM Configuration ----------
# Setup for the Ollama LLM using a local endpoint.
llm = OllamaLLM(
    base_url="http://10.38.187.118:11434",  # Ollama server URL
    model="gemma3",                         # model name
    temperature=0.0,                        # 0.0 makes the output deterministic
    callbacks=[callback_handler]           # Use custom logging callback
)


# ---------- Tool Functions ----------

# Calls a weather API to get current temperature and description for a city.
def get_weather(city: str) -> str:
    API_KEY = "my_key"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            return f"Error: {data.get('message', 'unknown error')}"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"{city.title()}: {temp}°C, {desc}"
    except Exception as e:
        return f"Error: {e}"

# Prepares a clean city name before passing it to get_weather.
def get_weather_cleaned(input_str: str) -> str:
    cleaned_city = input_str.strip()
    if cleaned_city.lower().startswith("city:"):
        cleaned_city = cleaned_city[5:].strip(" '\"")
    return get_weather(cleaned_city)

# Uses an IP geolocation API to find the city and country for a given IP address.
def ip_geolocation(ip: str) -> str:
    try:
        url = f"http://ip-api.com/json/{ip}"
        response = requests.get(url)
        data = response.json()
        if data["status"] != "success":
            return f"Error: {data.get('message', 'Unable to fetch IP info')}"
        return f"IP {ip} is located in {data['city']}, {data['regionName']}, {data['country']}. ISP: {data['isp']}."
    except Exception as e:
        return f"Error: {e}"

# Evaluates math expressions safely using only allowed math functions.
def compute_math_expression(expression: str) -> str:
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names["abs"] = abs
        allowed_names["round"] = round
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"The result of '{expression}' is {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

# Uses the LLM to answer general knowledge questions.
def answer_general_knowledge(question: str) -> str:
    try:
        return llm.invoke(question)
    except Exception as e:
        return f"Error: {e}"


# ---------- Register Tools ----------
# Each tool has a name, a function, and a description for agent usage.
tools = [
    Tool(
        name="WeatherTool",
        func=get_weather_cleaned,
        description="Use this to get the weather for a city. Input should be a city name and no other words."
    ),
    Tool(
        name="IPGeolocation",
        func=ip_geolocation,
        description="Returns location details of an IP address. Input should be a valid IP."
    ),
    Tool(
        name="MathTool",
        func=compute_math_expression,
        description="Use this to compute a math expression. Input should be a valid math expression."
    ),
    Tool(
        name="KnowledgeTool",
        func=answer_general_knowledge,
        description="Use this for answering general knowledge questions."
    )
]


# ---------- Prompt Template ----------
# Constructs a formatted prompt guiding the agent on how to reason and use tools.
tool_names = ", ".join([tool.name for tool in tools])
tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

react_prompt_template = f"""You are an intelligent agent helping answer user questions.

Use the available tools to answer user queries, even for general knowledge. Evaluate query and decide the best tool to use.

Use the following format to reason:

Question: {{input}}
Thought: I need to use a tool to answer this.
Action: <ToolName>
Action Input: <input>
Observation: <result>
... (repeat as needed)
Final Answer: <final answer>

### Example when general knowledge is requested, and KnowledgeTool tool is needed:

Question: What is the capital of Denmark?
Thought: I need to use a tool to answer this.
Action: KnowledgeTool
Action Input: What is the capital of Denmark?
Observation: The capital of Denmark is Copenhagen.
Final Answer: The capital of Denmark is Copenhagen.

Question: What is the boiling point of water?
Thought: I need to use a tool to answer this.
Action: KnowledgeTool
Action Input: What is the boiling point of water?
Observation: The boiling point of water is 100°C or 212°F at standard atmospheric pressure.
Final Answer: The boiling point of water is 100°C or 212°F at standard atmospheric pressure.

Question: What is the tallest mountain in the world?
Thought: I need to use a tool to answer this.
Action: KnowledgeTool
Action Input: What is the tallest mountain in the world?
Observation: Mount Everest is the tallest mountain in the world.
Final Answer: Mount Everest is the tallest mountain in the world.

Question: What is the currency of Japan?
Thought: I need to use a tool to answer this.
Action: KnowledgeTool
Action Input: What is the currency of Japan?
Observation: The currency of Japan is the Japanese Yen.
Final Answer: The currency of Japan is the Japanese Yen.


### Example when real-time weather is requested, and WeatherTool tool is needed:

Question: What is the weather in London?
Thought: I need to use WeatherTool tool to answer this.
Action: WeatherTool
Action Input: London
Observation: London: 15°C, clear sky
Final Answer: The weather in London is 15°C with clear sky.


### Example when math expression needs to be computed, and MathTool tool is needed:

Question: What is the result of sqrt(49) + log(100, 10)?
Thought: I need to use MathTool tool to answer this.
Action: MathTool
Action Input: sqrt(49) + log(100, 10)
Observation: The result of 'sqrt(49) + log(100, 10)' is 9.0
Final Answer: The result is 9.0.


### Example when IP geolocation is requested, and IPGeolocation tool is needed:

Question: Where is the IP address 8.8.8.8 located?
Thought: I need to use a tool to answer this.
Action: IPGeolocation
Action Input: 8.8.8.8
Observation: IP 8.8.8.8 is located in Ashburn, Virginia, United States. ISP: Google LLC.
Final Answer: The IP address 8.8.8.8 is located in Ashburn, Virginia, United States. The ISP is Google LLC.


### Available Tools

KnowledgeTool: Use this for answering general knowledge questions like capitals, history, science, famous people or other general knowledge learned during pretraining.
WeatherTool: Use this to get the weather for a city. Input should be a city name like 'Berlin'.
MathTool: Returns result of a given math expression. Input should be a valid IP like '100+450'.
IPGeolocation: Returns location details of an IP address. Input should be a valid IP like '18.81.81.81'.


Begin answering:
Question: {{input}}
{{agent_scratchpad}}"""

# Compiles the prompt into a usable LangChain PromptTemplate object.
react_prompt = PromptTemplate(
    template=react_prompt_template,
    input_variables=["input", "agent_scratchpad"]
)


# ---------- Agent Setup ----------
# Initialize the agent with tools, prompt, LLM, and other configurations.
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Agent type that reasons step-by-step
    verbose=True,                                 # Enables verbose output for debugging
    handle_parsing_errors=True,                   # Prevents crash if agent output is malformed
    callbacks=[callback_handler],                 # Attach logging
    agent_kwargs={"prompt": react_prompt},        # Use our custom prompt template
    max_iterations=5                              # Limit steps to prevent infinite loops
)


# ---------- API Endpoint ----------
# Defines a POST endpoint at /ask to receive a question and return an answer.
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    query = req.question
    logger.info(f"\n>>> [INPUT QUESTION] {query}")
    
    # Run the question through the agent
    result = agent.invoke({
        "input": query,
        "agent_scratchpad": ""
    })

    logger.info(f">>> [AGENT OUTPUT] {result}")
    logger.info(f">>> [FINAL ANSWER] {result.get('output')}")

    return {"question": query, "answer": result["output"]}


# ---------- Start Server ----------
# If the script is run directly, start the FastAPI app using Uvicorn.
if __name__ == "__main__":
    uvicorn.run("basic:app", host="0.0.0.0", port=8000, reload=True)
