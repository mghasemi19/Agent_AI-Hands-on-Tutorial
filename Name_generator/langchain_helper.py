import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set.")

if not base_url:
    raise ValueError("BASE_URL is not set.")


def generate_pet_name(animal_type: str, pet_color: str,) -> str:
    """Generate five pet-name suggestions."""

    if not animal_type.strip():
        raise ValueError("animal_type cannot be empty.")

    if not pet_color.strip():
        raise ValueError("pet_color cannot be empty.")

    chat_model = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model="gapgpt-qwen-3.5",
        temperature=0.7,
        max_tokens=400,
    )

    prompt = PromptTemplate.from_template(
        """
        I have a {animal_type} pet that is {pet_color}.
        Suggest exactly five cool names for my pet.
        Return only list number and names and not anything else.
        """.strip()
    )

    chain = prompt | chat_model | StrOutputParser()

    response = chain.invoke(
        {
            "animal_type": animal_type,
            "pet_color": pet_color,
        }
    )

    return response

'''
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from dotenv import load_dotenv

load_dotenv()


def generate_pet_name(animal_type, pet_color, openai_api_key):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

    prompt_template_name = PromptTemplate(
        input_variables = ['animal_type','pet_color'],
        template = "I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color. Suggest me five cool names for my pet."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")

    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})


    return response

if __name__ == "__main__":
    print(generate_pet_name("Dog", "Black"))
'''