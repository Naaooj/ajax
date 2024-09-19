from openai import AzureOpenAI

import constants as cs
import configparser
import os

class AzureService():
    """
    The AzureService class is responsible for calling the OpenAI Vision API to extract data from the images.

    For more information on the OpenAI client, see https://learn.microsoft.com/en-us/azure/ai-services/openai/references/on-your-data?tabs=python
    For more information on the OpenAI Vision API, see https://github.com/Azure/azure-rest-api-specs/blob/main/specification/cognitiveservices/data-plane/AzureOpenAI/inference/preview/2024-02-15-preview/inference.json
    """
    
    def __init__(self):
        """
        Instiantiate the AzureService class, loading the configuration from the config.ini file and creating an OpenAI client.
        """
        # Load configuration from config.ini
        self.__config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        self.__config.read(config_path)

        # Create an OpenAI client
        self.__client = AzureOpenAI(
            api_key=self.__config['openai']['key'],
            api_version=self.__config['openai']['api_version'],
            base_url=f"{self.__config['openai']['endpoint']}openai/deployments/{self.__config['openai']['deployment_name']}",
        )

    def call_vision_api(self, images):
        """
        Call the OpenAI Vision API to extract data from the images
        
        Parameters:
            images: The list of images in base64 format
        
        Returns:
            out: The response from the API
        """
        model = self.__config['openai']['deployment_name']

        # List holdings the user prompt parts (text and base64 images)
        user_prompt_parts = self.__initialize_user_images_prompt_parts(images)

        # List holding the messages (system and user prompts)
        messages = self.__create_messages(user_prompt_parts)

        return self.__client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=800, # Our free version of Azure subscribtion is limited to 1k per minutes, if we go above we are blocked for 24h
            temperature=0.1 # We want to have the most accurate result (0 = deterministic, 1 = default, 2 = very creative)
        )
    
    def __initialize_user_images_prompt_parts(self, images):
        """
        Initialize the user images prompt parts

        Parameters:
            images: The list of images in base64 format
        
        Returns:
            out: The list of user prompt corresponding to the images
        """
        user_prompt_parts = []
        for image in images:
            json_object = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}"
                }
            }
        
            # Add the JSON object to the list
            user_prompt_parts.append(json_object)

        return user_prompt_parts

    def __create_messages(self, user_prompt_parts):
        """
        Create the messages that will be sent to the OpenAI API
        
        Parameters:
            user_prompt_parts: The list of user prompt parts (text and base64 images)

        Returns: 
            out: The list of messages to be sent to the API
        """
        content = []
        content.append({
            "type": "text",
            "text": cs.USER_PROMPT
        })

        # Add the user prompt parts to the content list
        content.extend(user_prompt_parts)

        return [
            {
                "role": "system",
                "content": cs.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": content
            }
        ]