from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

promp1 = PromptTemplate(
    template = 'Generate the Sophisticated thing you can tell about it {topic}',
    input_vars = ['topic']
)

promp2 = PromptTemplate(
    template = 'Generate the best about the following {topic}',
    input_vars = ['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser  = StrOutputParser()

chain = promp1 | model | parser | promp2 | model | parser

print(chain.invoke({'topic':'RabindranathNath Thakur'}))