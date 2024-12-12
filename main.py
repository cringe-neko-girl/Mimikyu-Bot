import os
import sys
import subprocess
import traceback
import asyncio
import requests
from aiohttp import web
import time
from discord.ext import commands
from discord import HTTPException
from colorama import Fore, Style
from dotenv import load_dotenv
import pymongo
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConfigurationError

os.system('pip install --upgrade pip')

env_path = os.path.join('.github', '.env')
load_dotenv(dotenv_path=env_path)


from Imports.depend_imports import *
import Imports.depend_imports as depend_imports
from Imports.discord_imports import *
from Imports.log_imports import logger
from Cogs.pokemon import PokemonPredictor

class BotSetup(commands.AutoShardedBot):
    def __init__(self):
        intents = discord.Intents.all()
        intents.members = True
        super().__init__(command_prefix=commands.when_mentioned_or('...'),
                         intents=intents,
                         help_command=None,
                         shard_count=1, shard_reconnect_interval=10)
        self.mongoConnect = None

    async def on_ready(self):
        print(Fore.GREEN + f"Logged in as {self.user} (ID: {self.user.id})" + Style.RESET_ALL)

    
    async def start_bot(self):
        await self.setup()
        token = os.getenv('TOKEN')
        if not token:
            logger.error("No token found. Please set the TOKEN environment variable.")
            return
        try:
            await self.start(token)
        except KeyboardInterrupt:
            await self.close()
        except Exception as e:
            traceback_string = traceback.format_exc()
            logger.error(f"An error occurred while logging in: {e}\n{traceback_string}")
            await self.close()
        finally:
            if self.is_closed():
                print("Bot is closed, cleaning up.")
            else:
                print("Bot is still running.")
            await self.close()

    async def setup(self):
        print("\n")
        print(Fore.BLUE + "・ ── Cogs/" + Style.RESET_ALL)
        await self.import_cogs("Cogs")
        print("\n")
        print(Fore.BLUE + "・ ── Events/" + Style.RESET_ALL)
        await self.import_cogs("Events")
        print("\n")
        print(Fore.BLUE + "===== Setup Completed =====" + Style.RESET_ALL)

    async def import_cogs(self, dir_name):
        files_dir = os.listdir(dir_name)
        for filename in files_dir:
            if filename.endswith(".py"):
                print(Fore.BLUE + f"│   ├── {filename}" + Style.RESET_ALL)
                module = __import__(f"{dir_name}.{os.path.splitext(filename)[0]}", fromlist=[""])
                for obj_name in dir(module):
                    obj = getattr(module, obj_name)
                    if isinstance(obj, commands.CogMeta):
                        if not self.get_cog(obj_name):
                            await self.add_cog(obj(self))
                            print(Fore.GREEN + f"│   │   └── {obj_name}" + Style.RESET_ALL)

async def check_rate_limit():
    url = "https://discord.com/api/v10/users/@me"
    headers = {
        "Authorization": f"Bot {os.getenv('TOKEN')}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        remaining_requests = int(response.headers.get("X-RateLimit-Remaining", 1))
        rate_limit_reset_after = float(response.headers.get("X-RateLimit-Reset-After", 0))
        if remaining_requests <= 0:
            logger.error(f"Rate limit exceeded. Retry after {rate_limit_reset_after} seconds.")
            print(f"Rate limit exceeded. Please wait for {rate_limit_reset_after} seconds before retrying.")
            await asyncio.sleep(rate_limit_reset_after)
    else:
        logger.error(f"Failed to check rate limit. Status code: {response.status_code}")

async def main():
    bot = BotSetup()
    try:
        await check_rate_limit()
        await bot.start_bot()
    except HTTPException as e:
        if e.status == 429:
            retry_after = int(e.response.headers.get("Retry-After", 0))
            logger.error(f"Rate limit exceeded. Retry after {retry_after} seconds.")
            print(f"Rate limit exceeded. Please wait for {retry_after} seconds before retrying.")
            await asyncio.sleep(retry_after)
        else:
            traceback_string = traceback.format_exc()
            logger.error(f"An error occurred: {e}\n{traceback_string}")
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"An error occurred: {e}\n{traceback_string}")
    finally:
        await bot.close()

async def start_http_server():
    try:
        app = web.Application()
        app.router.add_get('/', lambda request: web.Response(text="Bot is running"))
        runner = web.AppRunner(app)
        await runner.setup()
        port = int(os.getenv("PORT", 8080))
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        print(f"HTTP server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start HTTP server: {e}")
        print("Failed to start HTTP server.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(start_http_server())
    loop.run_until_complete(main())
