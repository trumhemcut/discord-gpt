from config import settings

from langchain.llms import OpenAI
from langchain.agents import load_tools, get_all_tool_names, ConversationalAgent
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

from .loader import load_pdf_documents, load_txt_documents

llm = OpenAI(temperature=0, openai_api_key=settings.OPENAI_API_KEY)
zapier = ZapierNLAWrapper(zapier_nla_api_key=settings.ZAPIER_NLA_API_KEY)
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
zapier_tools = toolkit.get_tools()
ddg_search = DuckDuckGoSearchAPIWrapper()

for tool in zapier_tools:
    if 'send_email' in tool.zapier_description:
        tool.description = f"""useful for when you need to send email to someone. You need to provide the email address (in correct format), subject, and the body of the email. The format should be: "email": "the email's details", "subject": "the subject's details", "body": "the body's details". You provide the details in the body so that the receiver understand the content. The more details, the better!"""
    else:
        tool.description = f'useful for when you need to do: {tool.zapier_description}'

ddg_tool = Tool(
    name='DuckDuckGo Search',
    func=ddg_search.run,
    description="useful for when you need to search latest song tracks on Spotify, or search for the latest news and events. Input should be a search query.",
)

pdf_docs_tool = Tool(
    name="pdf_tool",
    func=load_pdf_documents(llm=llm),
    description="useful for when you need to answer any questions about our company (Contoso Electronics) such as employee handbook, company mission, values, performance reviews and other company's information. Input should be a fully formed question.",
)

txt_docs_tool = Tool(
    name="docs_tool",
    func=load_txt_documents(llm=llm),
    description="useful for when you search internal documents (such as Phi Huynh's birthday). Input should be a fully formed question.",
)


tools = []

tools.append(ddg_tool)
tools.append(txt_docs_tool)
tools.append(pdf_docs_tool)
tools = tools + zapier_tools

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, max=3, memory=memory)


class Completion:
    def __init__(self, agent):
        self.agent = agent

    def response(self, query: str):
        return self.agent.run(query)


completion = Completion(agent)
