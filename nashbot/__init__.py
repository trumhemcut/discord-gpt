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
    elif 'create_event_reminder' in tool.zapier_description:
        tool.description = f"""useful for when you need to create an event or set a reminder in Outlook. You need to provide subject, start date & time, end date & time. The format should be: "subject": "the subject's details", "start_datetime": "the start date time", "end_datetime": "the end date time". """
    elif 'chat' in tool.zapier_description:
        tool.description = f"""useful for when you need to send message to someone via Microsoft Teams. You need to provide chat, message_text fields. The format should be: "chat": "the person you want to chat", "message_text": "the chat content". Always add "sent_by_discord" at the end of message_text."""
    elif 'create_playlist' in tool.zapier_description:
        tool.description = f"""useful for when you need to create a playlist on Spotify. You need to provide playlist_name field. The format should be: "playlist_name": "the name of the playlist"."""
    elif 'add_track' in tool.zapier_description:
        tool.description = f"""useful for when you need to add a track to a playlist on Spotify. You need to provide playlist, and track__uri fields. For example: "playlist": "Only You", "track__uri": "spotify:track:2ccW4vFSVKRgVjkZzvdjRw"."""
    elif 'find_track' in tool.zapier_description:
        tool.description = f"""useful for when you need to search or find a track on Spotify. You need to provide search query field which include song title and artist. For example: "search_query": "hound dog elvis presley"."""
    else:
        tool.description = f'useful for when you need to do: {tool.zapier_description}'

ddg_tool = Tool(
    name='ddg',
    func=ddg_search.run,
    description="useful for when you need to search for the latest news and events. Input should be a search query.",
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
    tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, max=10, memory=memory)


class Completion:
    def __init__(self, agent):
        self.agent = agent

    def response(self, query: str):
        return self.agent.run(query)


completion = Completion(agent)
