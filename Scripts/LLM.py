import logging
from dotenv import load_dotenv
import openai
import os
from pydantic import BaseModel

def call_GPT(
    system_prompt: str,
    user_query: str,
    pydantic_model: BaseModel,
    model: str = "gpt-4o-mini",
) -> BaseModel:
    """
    Calls the OpenAI GPT model with the given system prompt and user query,
    and parses the response using the provided Pydantic model.

    Args:
        system_prompt (str): The system prompt for GPT.
        user_query (str): The user query to send to GPT.
        pydantic_model (BaseModel): The Pydantic model to parse the response.
        model (str): The GPT model to use.

    Returns:
        BaseModel: The parsed response as a Pydantic model.
    """
    try:
        if model == "gpt-4o" or model == "gpt-4o-mini":
            response = openai.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                response_format=pydantic_model,
            )

            parsed_response = response.choices[0].message.parsed
            return parsed_response

        elif model == "o3-mini":
            response = openai.beta.chat.completions.parse(
                model=model,
                reasoning_effort="high",
                messages=[
                    {"role": "user",
                     "content": f"{system_prompt} ; {user_query}"},

                ],
                response_format=pydantic_model,
            )

            parsed_response = response.choices[0].message.parsed
            return parsed_response

    except Exception as e:
        print(f"Error generating LLM response: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Load environment variables from .env file
    load_dotenv()

    # Access the OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.critical("No OpenAI API key found in environment variables.")
        raise ValueError("No OpenAI API key found in environment variables.")

    openai.api_key = api_key
    logger.info("OpenAI API key loaded successfully.")
