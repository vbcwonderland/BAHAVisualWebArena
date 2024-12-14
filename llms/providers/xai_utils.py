import os
from typing import List
import openai
from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")

if not XAI_API_KEY:
    raise EnvironmentError("XAI_API_KEY is not set or is inaccessible.")

print("XAI_API_KEY:", XAI_API_KEY)


client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

def generate_from_grok_vision_url(
    text_prompt: str,
    image_urls: List[str],
    model: str = "grok-vision-beta",
    temperature: float = 0.01,
) -> str:
    """
    Generate content from xAI's grok-vision-beta model using text and image URLs.

    Args:
        text_prompt: Text prompt describing the task or question.
        image_urls: List of URLs to images.
        model: Model name to use (default: grok-vision-beta).
        temperature: Sampling temperature for response diversity.

    Returns:
        Generated response as a string.
    """
    # Prepare messages with text and image URLs
    content = [
        {"type": "image_url", "image_url": {"url": url, "detail": "high"}}
        for url in image_urls
    ]
    content.append({"type": "text", "text": text_prompt})

    messages = [{"role": "user", "content": content}]

    # Send the request to xAI
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=temperature,
    )

    # Aggregate and return the response
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content
    return response

# Example Usage
if __name__ == "__main__":
    text_prompt = "Describe this image?"
    image_urls = [
        "https://x.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Faurora.577c2a9b.png&w=1080&q=75",
    ]

    try:
        # Generate response using grok-vision-beta
        result = generate_from_grok_vision_url(
            text_prompt=text_prompt,
            image_urls=image_urls
        )
        print("Generated Response:", result)
    except Exception as e:
        print(f"An error occurred: {e}")
