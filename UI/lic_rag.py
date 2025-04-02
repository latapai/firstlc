import chainlit as cl
import requests
import asyncio

import os
from dotenv import load_dotenv
load_dotenv()


FASTAPI_ENDPOINT = os.getenv("BACKEND_URL") + "/call_watsonx"


@cl.on_message
async def on_message(message: cl.Message):
    
    #image = cl.Image(path="C:/LIC_RAG_2024DEC/UI/1-05-wo-hat.png", name="image1", display="inline")

    # Attach the image to the message
    #await cl.Message(
    #    content="I am your assistant....Hello",
    #    elements=[image],
    #).send()

    # Access the content of the message (i.e., the user's input as a string)
    
    user_input = message.content
    
    # Send a "Typing..." message to indicate loading
    loading_message = await cl.Message(content="Fetching...").send()
    
    # Send the user's question to the backend
    response = requests.post(FASTAPI_ENDPOINT, json={"question": user_input})
    
    # Extract the generated response from the API response JSON
    generated_response = response.json().get("response", "No response found.")
    
    # Edit the loading message with the actual response
    loading_message.content = generated_response
    await loading_message.update()

# Run the app
if __name__ == "__main__":
    cl.run()
