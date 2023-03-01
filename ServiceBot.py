from flask import Flask, request
import requests
from twilio.twiml.messaging_response import MessagingResponse
import os
from dotenv import load_dotenv
from pathlib import Path
from gpt_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, GPTListIndex, GPTIndexMemory
from langchain.prompts import BasePromptTemplate
from pydantic import BaseModel, validator

from langchain import OpenAI
from langchain.agents import initialize_agent

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferMemory

from langchain.chains import ConversationChain



class StavrosPromptTemplate(BasePromptTemplate, BaseModel):
    """ A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function. """

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """ Validate that the input variables are correct. """
        if "question" not in v:
            raise ValueError("question must be an input_variable.")
        return v

    def format(self, **kwargs)->str:
        chat_history = ""
        # Get the source code of the function
        #load bio from file bio.txt
        with open('bio.txt', 'r') as file:
            bio = file.read().replace('\n', '')
        # Generate the prompt to be sent to the language model
        prompt = f"""
        The following bio describes who you are:
        Bio:
        {bio}
        End Bio:
        Also consider the following interaction between you and a human when answering the question below:
        Chat History:{kwargs["chat_history"]}
        Question:{kwargs["question"]}
        
        Answer the question as Stavros
        """

        return prompt

    def _prompt_type(self):
        return "service-bot"

    def responded(self):
        return True


app = Flask(__name__)
responded = False


@app.route('/bot', methods=['POST'])
# out of the box tutorial from Twilio
@app.route('/ServiceBotNoMem', methods=['POST'])
def stavros_nomem():
    incoming_msg = request.values.get('Body', '').lower()
    #create response object
    resp = MessagingResponse()
    msg = resp.message()
    responded = False
    #call index to query
    chat_history=""
    StavrosPrompt = StavrosPromptTemplate(input_variables=["question","chat_history"])
    prompt = (StavrosPrompt.format(question=incoming_msg,chat_history=chat_history))
    response = index.query(prompt)
    # print to show in terminal
    print(incoming_msg)
    msg.body(str(response))
    print(response)
    responded = True
    return str(resp)



@app.route('/ServiceBot', methods=['POST'])
def stavros():
    global responded
    global agent_chain
    global promp

    # process incoming message
    incoming_msg = request.values.get('Body', '').lower()

    # create response object
    resp = MessagingResponse()
    msg = resp.message()
    chat_history = ""
    if responded is not True:
        app.logger.info("First request, initializing agent chain")
        prompt = (StavrosPrompt.format(question=incoming_msg, chat_history=chat_history))
        app.logger.info(prompt)

        tools = [
            Tool(
                name="GPT Index",
                func=lambda q: str(index.query(q)),
                description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
                return_direct=True
            ),
        ]
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm = OpenAI(temperature=0.5)
        agent_chain = initialize_agent(tools, llm, agent="conversational-react-description",memory=memory)
        #agent_chain = initialize_agent([], llm=OpenAI(temperature=.5), prompt=prompt,agent="zero-shot-react-description")
    if responded is True:
        app.logger.info("Subsequent Request, updating prompt")
        prompt = (StavrosPrompt.format(question=incoming_msg, chat_history=chat_history))
        app.logger.info(prompt)
    response = agent_chain.run(input=prompt)
    app.logger.info(str(agent_chain))
    msg.body(str(response))
    responded = True

    # print to show in terminal
    print(incoming_msg)
    print(resp)
    return str(resp)


@app.route('/', methods=['GET'])
def index():
    return 'Web interface for ServiceBot coming soon!'
    # expand to web UI
    # return render_template('index.html', messages=["hi", "hello"])


if __name__ == '__main__':
    # load index
    global prompt
    global chat_history
    dotenv_path = Path('CRED.env')
    load_dotenv(dotenv_path=dotenv_path)
    os.getenv('OPENAI_API_KEY')

    chat_history = [""]
    prompt = ""

    # index the contex
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    #memory = GPTIndexMemory(index=index, memory_key="chat_history", query_kwargs={"response_mode": "compact"})
    StavrosPrompt = StavrosPromptTemplate(input_variables=["question", "chat_history"])

    # run the app
    app.run(host="localhost", port=5003, debug=True)