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


class StavrosPromptTemplate(BasePromptTemplate, BaseModel):
    """ A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function. """

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """ Validate that the input variables are correct. """
        if "question" not in v:
            raise ValueError("question must be an input_variable.")
        return v

    def format(self, **kwargs) -> str:
        chat_history = ""
        # Get the source code of the function

        # Generate the prompt to be sent to the language model
        prompt = f"""
        You are a chatbot named Stavros having a conversation with a human customer who asks questions, and it's important that you remember the entire discussion in case you're asked to summarize all of it.
        First,introduce yourself as an old Greek grandpa named Stavros, excited to have people experience your Greek culture at Little Greek Fresh Grill.
        Answer all human questions concisely; incorporate your childhood stories in Mykonos when possible.
        You can also tell them about your favorite Greek foods (Feta Cheese, Pastitsio and Lamb Gyro are some but you can make up others), and how you make them. Keep responses to under 4 sentences.
        Offer a promotion after 3 questions, providing the human a promo code 'Stavros20'.
        
        Here is an example of an exchange:
        Human: Tell me about your menu.
        AI: Hi, my name is Stavros, and I'm excited to have you experience our Greek culture at Little Greek Fresh Grill. Our menu features a variety of Greek-inspired dishes, including gyros, salads, and wraps. We also offer a selection of sides, desserts, and drinks. All of our menu items are made with fresh ingredients and are prepared with care. We strive to provide our customers with a unique and delicious dining experience.
        Human: Great. Do you have gluten-free options?
        AI: Yes, we offer a variety of gluten-free options. Our menu includes grilled chicken, fish, and salads, as well as a selection of gluten-free sides, such as roasted potatoes, rice, and quinoa.
        Human: Sounds good. How about dairy-free options?
        AI: Yes, we do offer dairy free options. Our menu includes a variety of vegan and vegetarian dishes, such as falafel, hummus, and tabouli. We also offer a selection of vegan and vegetarian sides, such as roasted vegetables, quinoa, and couscous.
        Human: Thank you. Can you summarize our discussion?
        AI: You are very welcome! We discussed the menu and restaurant that represents are deep Greek culture. I answered your questions about our availabilty of gluten-free and dairy-free options just like you asked. Please come to the restaurant, and enjoy 20% off using promo code 'Stavros20'
        
        {chat_history}
        Question:{kwargs["question"]}
        AI:
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
def bot():
    # how to process message
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    responded = False
    # opentional logic
    if 'quote' in incoming_msg:
        # return a quote
        r = requests.get('https://api.quotable.io/random')
        if r.status_code == 200:
            data = r.json()
            quote = f'{data["content"]} ({data["author"]})'
        else:
            quote = 'I could not retrieve a quote at this time, sorry.'
        msg.body(quote)
        responded = True
    if 'cat' in incoming_msg:
        # return a cat pic
        msg.media('https://cataas.com/cat')
        responded = True
    if not responded:
        msg.body('I only know about famous quotes and cats, sorry!')
    return str(resp)


@app.route('/ServiceBot', methods=['POST'])
def stavros():
    global responded
    global agent_chain
    # process incoming message
    incoming_msg = request.values.get('Body', '').lower()

    # create response object
    resp = MessagingResponse()

    StavrosPrompt = StavrosPromptTemplate(input_variables=["question", "chat_history"])
    prompt = (StavrosPrompt.format(question=incoming_msg))
    if responded is not True:
        agent_chain = initialize_agent([], llm=OpenAI(temperature=0.3), prompt=prompt, input="question",
                                   agent="conversational-react-description", memory=memory)
    resp = agent_chain.run(incoming_msg)
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
    dotenv_path = Path('CRED.env')
    load_dotenv(dotenv_path=dotenv_path)
    os.getenv('OPENAI_API_KEY')

    # index the contex
    index = GPTSimpleVectorIndex.load_from_disk('index.json')

    # define the conversational memory for the service bot
    memory = GPTIndexMemory(index=index, memory_key="chat_history", query_kwargs={"response_mode": "compact"})

    # run the app
    app.run(host="localhost", port=5003, debug=True)