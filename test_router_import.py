try:
    from vanna.servers.fastapi import VannaRouter
    print("Success: from vanna.servers.fastapi import VannaRouter")
except ImportError:
    print("Failed: from vanna.servers.fastapi import VannaRouter")

try:
    from vanna.servers.base import VannaRouter
    print("Success: from vanna.servers.base import VannaRouter")
except ImportError:
    print("Failed: from vanna.servers.base import VannaRouter")
