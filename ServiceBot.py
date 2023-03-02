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

        history_blob="".join(kwargs["chat_history"])
        # Generate the prompt to be sent to the language model
        prompt = f"""
        The following bio describes who you are:
        Bio:
        {bio}
        End Bio:
        Also consider the following interaction between you and a human when answering the question below:
        Chat History:{history_blob}
        Question:{kwargs["question"]}
        
        Answer the question as Stavros considering the chat history and bio above, try to keep the response under 3 sentences:
        """

        return prompt

    def _prompt_type(self):
        return "service-bot"

    def responded(self):
        return True


app = Flask(__name__)
responded = False

@app.route('/ServiceBotNoMem', methods=['GET','POST'])
def stavros_nomem():
    chat_history = ""
    if request.method == 'POST':
        incoming_msg = request.values.get('Body', '').lower()
        #create response object
        resp = MessagingResponse()
        msg = resp.message()
        responded = False
        #call index to query

        StavrosPrompt = StavrosPromptTemplate(input_variables=["question","chat_history"])
        prompt = StavrosPrompt.format(question=incoming_msg,chat_history=chat_history)
        response = index.query(prompt)
        # print to show in terminal
        print(incoming_msg)
        msg.body(str(response))
        print(response)
        responded = True
        return str(resp)
    else:
        static_prompt = "Tell me about yourself?"
        # create response object
        input_prompt = request.args.get('prompt')
        if input_prompt:
            prompt = input_prompt
        else:
            prompt= static_prompt
        # call index to query
        StavrosPrompt = StavrosPromptTemplate(input_variables=["question","chat_history"])
        prompt = StavrosPrompt.format(question=prompt,chat_history=chat_history)
        response = index.query(prompt)
        # print to show in terminal
        print(static_prompt)
        print(response)
        responded = True
        return str(response)



@app.route('/ServiceBot', methods=['POST'])
def stavros():
    global responded
    global agent_chain
    global promp

    # Process incoming message
    incoming_msg = request.values.get('Body', '').lower()
    promo = ""

    # If mykonos is in incoming message add promo code
    if "mykonos" in (incoming_msg):
        promo = "\nI love talking about Mykonos - Use promo code: mykonos20 for 20% off your next online order!"

    # create response object
    resp = MessagingResponse()
    msg = resp.message()

    # First Time Only - ServiceBot chat invoked - create / initialize agent chain
    if responded is not True:
        app.logger.info("First request, initializing agent chain")
        prompt = (StavrosPrompt.format(question=incoming_msg, chat_history=chat_history))
        app.logger.info(prompt)

        tools = [
            Tool(
                name="GPT Index",
                func=lambda q: str(index.query(q)),
                description="Answer questions about the restaurant using pre-indexed content. The input to this tool should be a complete english sentence.",
                return_direct=True
            ),
        ]

        llm = OpenAI(temperature=0.5)
        agent_chain = initialize_agent(tools, llm, agent="conversational-react-description",memory=memory)

    # Post-first chat message handling - maintain updates to the prompt to include up-to-date chat history
    if responded is True:
        app.logger.info("Subsequent Request, updating prompt")
        app.logger.info(chat_history)
        prompt = (StavrosPrompt.format(question=incoming_msg, chat_history=chat_history))
        app.logger.info(prompt)

    # Generate a response based upon current, cumulative interaction between human and bot
    app.logger.info("PROMO:" + promo)
    response = str(agent_chain.run(input=prompt)) + promo
    chat_history.append("\nHuman:"+incoming_msg)
    chat_history.append("\nStavros:" + response)
    app.logger.info("\nCHAT HISTORY:\n"+str(chat_history))
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
    global memory

    memory = ConversationBufferMemory(memory_key="chat_history")
    dotenv_path = Path('/Users/ksimon00/.zshrc')
    load_dotenv(dotenv_path=dotenv_path)
    os.getenv('OPENAI_API_KEY')

    chat_history = [""]
    prompt = ""

    # index the context
    index = GPTSimpleVectorIndex.load_from_disk('index.json')

    # define the prompt, using the predefined template completed with instruction and dyanmic bio information
    StavrosPrompt = StavrosPromptTemplate(input_variables=["question", "chat_history"])

    # run the app
    app.run(host="localhost", port=5003, debug=True)