import asyncio
import os
import time
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.contents import ChatHistory
from credentials.azureopenai import add_azure_openai_env_variables
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from tools.sk_websearch import BingWebSearchPlugin
from tools.sk_email_search import EmailSearchPlugin
from tools.sk_send_email import SendEmailPlugin
from tools.sk_time import DateTimePlugin


async def main(user_input:str, history, log:bool=False):
    add_azure_openai_env_variables()

    kernel = Kernel()

    chat_service =   AzureChatCompletion(
        api_key=  os.environ["AZURE_OPENAI_KEY"],
        endpoint= os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
    )
    kernel.add_service(chat_service)

    # kernel.add_plugin(BingWebSearchPlugin(), plugin_name="BingWebSearchPlugin")
    # kernel.add_plugin(EmailSearchPlugin(), plugin_name="EmailSearchPlugin")
    # kernel.add_plugin(SendEmailPlugin(), plugin_name="SendEmailPlugin")
    # kernel.add_plugin(DateTimePlugin(), plugin_name="TimePlugin")
    
    history.add_user_message(user_input)

    chat_completion : AzureChatCompletion = kernel.get_service(type=ChatCompletionClientBase)
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    response = (await chat_completion.get_chat_message_contents(
                chat_history=history,
                kernel=kernel,
                settings=execution_settings,
                arguments=KernelArguments(),
            ))[0]

    if log:
        print(str(response))

    return response

    

if __name__ == "__main__":
    history = ChatHistory()
    history.add_system_message("You are an AI agent who can help find information on the web and send emails.")
    while True:
        input_str = input("Please enter your input: ")
        loop = asyncio.new_event_loop()
        response = loop.run_until_complete(main(input_str,history))
        history.add_assistant_message(response.content)
        print(response.content)
