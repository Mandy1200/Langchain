from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

parser = StrOutputParser()

class feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object = feedback)

prompt1 = PromptTemplate(
    template = "classify the sentiment following feedback into positive or negative{feedback} \n {format_instruction}",
    input_vars = ['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = "write the best and heart melting feedback for the positive sentiment : {feedback}",
    input_vars = ['feedback']
)

prompt3 = PromptTemplate(
    template = "write the best and heart melting feedback for the negative sentiment : {feedback}",
    input_vars = ['feedback']
)

brain_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive" , prompt2 | model | parser),
    (lambda x:x.sentiment == "negative" , prompt3 | model | parser),
    RunnableLambda(lambda x : "Could Not Find Any sentiment")
)

chain = classifier_chain | brain_chain

print(chain.invoke({'feedback':'neither positive nor negative'}))