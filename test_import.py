try:
    from vanna.openai import OpenAI_Chat
    print("Success: from vanna.openai import OpenAI_Chat")
except ImportError:
    print("Failed: from vanna.openai import OpenAI_Chat")

try:
    from vanna.openai.openai_chat import OpenAI_Chat
    print("Success: from vanna.openai.openai_chat import OpenAI_Chat")
except ImportError:
    print("Failed: from vanna.openai.openai_chat import OpenAI_Chat")
