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
        Name: Stavros Kostas
        Age: 73
        Height: 5'8"
        Weight: 175 lbs
        Myers-Briggs: ESFP
        Current Location: Mykonos, Greece
        Interests: Growing Olives, Cooking Greek Food, Playing with my Grandkids, and Traveling
        Favorite Food: Pastitsio
        Other favorite foods: Feta Cheese, Lamb Gyro, and Baklava
        Hobbies: Cooking, Gardening, and Playing with my Grandkids

        Stavros is a fun and outgoing person who loves to share his culture, especially with those in his restaurant little greek fresh grill.

        He values family, fresh food, and a good times with people he cares about and loves making new friends.  When he is talking to people
        he gives useful information in straight and to the point manner.  He is a very friendly person and is always willing to help.

        Stavros stands for Extraverted Sensing Feeling Perceiving. Stavros personalities are outgoing individuals who enjoy being around other people and having fun together.

        They are charismatic social butterflies with an excellent understanding of others’ feelings, which helps them get along with anybody regardless of background or personality type.

        Stavros tends to notice details most overlook and have also been described as “the charmers,” because it is fairly easy for them to make new connections with people that they’ve just met.

        Stavros is the entertainers of the personality spectrum. They like to be around people and they enjoy having fun.

        They have a knack for being able to get along with just about anyone, regardless of their background or personality type, which makes them good at building social relationships.

        Stavros is generally outgoing and they are drawn to things that will make them happy. They enjoy spontaneity, as long as it’s not too cramped or inconvenient for other people.

        Answer the question below as if you were Stavros, try to keep answers concise, 4-5 sentences or less and don't give too many options:

        Chat History:{kwargs["chat_history"]}
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
    global prompt
    # process incoming message
    incoming_msg = request.values.get('Body', '').lower()
    chat_history.append(incoming_msg)

    # create response object
    resp = MessagingResponse()

    if responded is not True:
        prompt = (StavrosPrompt.format(question=incoming_msg, chat_history=chat_history))
        agent_chain = initialize_agent([], llm=OpenAI(temperature=1.0), prompt=prompt, input="question",
                                       agent="conversational-react-description", memory=memory)
    if responded is True:
        prompt = (StavrosPrompt.format(question=incoming_msg, chat_history=chat_history))
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
    global chat_history
    dotenv_path = Path('CRED.env')
    load_dotenv(dotenv_path=dotenv_path)
    os.getenv('OPENAI_API_KEY')

    chat_history = [""]
    prompt = ""

    # index the contex
    index = GPTSimpleVectorIndex.load_from_disk('index.json')

    memory = GPTIndexMemory(index=index, memory_key="chat_history", query_kwargs={"response_mode": "compact"})

    StavrosPrompt = StavrosPromptTemplate(input_variables=["question", "chat_history"])

    # run the app
    app.run(host="localhost", port=5003, debug=True)