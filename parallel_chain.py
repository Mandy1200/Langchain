from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model='gemini-2.5-pro')
model2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


prompt1 = PromptTemplate(
    template = 'Generate A Paragraph for {text} not in more than 60 words',
    input_vars = ['text']
)

prompt2 = PromptTemplate(
    template = 'Generate A Quiz for {text}',
    input_vars = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the both Paragraph and quiz into single document notes ->{notes} and {quiz}',
    input_vars = ['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain

text = ''' Virat Kohli is a legendary Indian cricketer and former captain of the national team. Widely regarded as one of the greatest batsmen of his era, he's known for his aggressive batting style, incredible consistency, and impeccable fitness. Kohli holds numerous records, including the most centuries in One Day International (ODI) cricket, surpassing Sachin Tendulkar. He led India to unprecedented success in Test cricket, and his relentless pursuit of excellence has made him a global icon. A member of the 2011 Cricket World Cup-winning and 2024 T20 World Cup-winning squads, he is celebrated for his ability to perform under pressure, particularly in run-chases. '''

result =final_chain.invoke({'text':text})

print(result)