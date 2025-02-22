from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import MessageGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv


load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

topic_model = ChatOpenAI(temperature=0.7)
blog_model = ChatOpenAI(temperature=0.9)

def generate_title(state):
    """Generates a blog post title based on user input."""
    prompt = f"Generate a creative and engaging blog post title for the topic: {state['messages'][-1].content}"
    title = topic_model.invoke([HumanMessage(content=prompt)])  # Use HumanMessage
    return {"messages": [title]}

def generate_blog_content(state):
    """Generates a detailed blog post based on the generated title."""
    title = state['messages'][-1].content
    prompt = f"Write a detailed and well-structured blog post based on the title: {title}"
    blog_content = blog_model.invoke([HumanMessage(content=prompt)])  # Use HumanMessage
    return {"messages": [blog_content]}


def make_blog_graph():
    """Creates a two-step agent workflow for blog generation."""
    graph_workflow = StateGraph(State)
    
    graph_workflow.add_node("title_generator", generate_title)
    graph_workflow.add_node("blog_writer", generate_blog_content)
    
    graph_workflow.add_edge("title_generator", "blog_writer")
    graph_workflow.add_edge("blog_writer", END)
    graph_workflow.add_edge(START, "title_generator")
    
    return graph_workflow.compile()

def make_alternative_blog_graph():
    """Alternative blog generation workflow with additional flexibility."""
    graph_workflow = StateGraph(State)
    
    def call_title_generator(state):
        return {"messages": [topic_model.invoke([HumanMessage(content=f"Create a blog title for: {state['messages'][-1].content}")])]}  
    
    def call_blog_writer(state):
        return {"messages": [topic_model.invoke([HumanMessage(content=f"Create a blog post for title: {state['messages'][-1].content}")])]} 
    
    graph_workflow.add_node("title_generator", call_title_generator)
    graph_workflow.add_node("blog_writer", call_blog_writer)
    
    graph_workflow.add_edge("title_generator", "blog_writer")
    graph_workflow.add_edge("blog_writer", END)
    graph_workflow.add_edge(START, "title_generator")
    
    return graph_workflow.compile()

blog_agent = make_blog_graph()
alternative_blog_agent = make_alternative_blog_graph()

