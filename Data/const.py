# Imports
import os
import io
import json
import traceback
from datetime import datetime
from PIL import Image

from Imports.discord_imports import *
import platform

# Constants
class AnyaImages:
    shocked_anya = "https://img-03.stickers.cloud/packs/20d46227-dcb0-4583-8d66-ee78d4743129/webp/a65e28be-a5fd-4654-8e7d-736dbd809df2.webp"
    awake_anya = 'https://media.tenor.com/9kLYJilshNMAAAAe/spy-x-family-anya.png'
    question_anya = 'https://i.pinimg.com/236x/b7/23/1f/b7231fbf87eee22b6d1f35f83e9a80bd.jpg'
    ping_banner_anya = 'https://i.pinimg.com/564x/db/98/ff/db98ffc40d53378a9999528b69d66d00.jpg'
    sleepy_anya = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR9y_MT3QHxXZVVzVlA94oFM8uIN0QH1fdw8Q6inySFmQ&s'
    new_mission_anya = 'https://i.pinimg.com/236x/b5/90/49/b590497e5e776909274ba40b040bba8c.jpg'
    ping_thumbnail = 'https://i.pinimg.com/236x/5d/d7/d1/5dd7d1d91933d59b8f21732efba70368.jpg'
    help_thumbnail = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQggKtOcHg_2xDGNeqtciU_-7iZ19F3lFbV091fGKq7KtJI5MuOzLBEPg2rQRZ9ru-tDGY&usqp=CAU'
    
# Embed Avatar
class EmbedFactory:
    @staticmethod
    async def change_avatar_prompt():
        embed = discord.Embed(
            title="Change Avatar",
            description="Please provide the image attachment or the URL to replace the avatar.\n\n"
                        "Type 'c' to cancel.",
            color=discord.Color.blue()
        )
        return embed

    @staticmethod
    async def successful_avatar_change(emoji_filename):
        embed = discord.Embed(
            title="Avatar Change Successful",
            description=f"The avatar has been successfully changed to `{emoji_filename}`.",
            color=discord.Color.green()
        )
        return embed

    @staticmethod
    async def failed_avatar_change():
        embed = discord.Embed(
            title="Avatar Change Failed",
            description=f"Failed to change the avatar.",
            color=discord.Color.red()
        )
        return embed

class Help_Embed_Mapping:
    embed = {
            "title": "Help Command",
            "description": "Need some help using certain commands? Take a look at the select options and choose the cog you need assistance with.",
            "thumbnail_url": AnyaImages.help_thumbnail,
            "image_url": "https://static1.cbrimages.com/wordpress/wp-content/uploads/2022/11/Spy-x-family-ep-18-Loid-and-Anyas-test-grades.jpg"
    }
    
class Help_Select_Embed_Mapping:
    embeds = {
        "system": {
            "title": "System",
            "description": "View the status of the bot, including information about its current performance, system resources usage, and configuration settings.",
            "color": discord.Color.red(),  # Customize color as needed
            "thumbnail_url": "https://i.pinimg.com/564x/f4/38/ef/f438ef92875df915c8d019780a76a346.jpg",
            "image_url": "https://i.pinimg.com/564x/f4/38/ef/f438ef92875df915c8d019780a76a346.jpg"
        },
        "quest": {
            "title": "Quest",
            "description": "```Still in Testing```\n**Estimated Time:** <t:1717165320:R>",
            "color": None,  # Customize color as needed
            "thumbnail_url": AnyaImages.sleepy_anya
        },
        
        "Cog2": {
            "title": "Title for Cog2",
            "description": "Description for Cog2",
            "color": discord.Color.blue(),  # Customize color as needed
            "thumbnail_url": "Thumbnail URL for Cog2"
        },
        # Add more embed mappings for other cogs as needed
    }

    emojis = {
        "system": "<:system_icon:1238536111266201610>",
        "quest": "<:loading_icon:1238532154292437022>",
        # Add more emoji mappings for other cogs as needed
    }
    
class Quest_Progress:
    # Dictionary to store progress bar mappings
    Progress_Bar_MAPPING = {
        "front_empty": 1237456749674364969,
        "front_full": 1237456617776218205,
        "mid_empty": 1237456613527523328,
        "mid_full": 1237456614697730139,
        "back_empty": 1237456616119599164,
        "back_full": 1237456619143696395
    }

    # Function to generate progress bar emojis
    @staticmethod
    async def generate_progress_bar(progress, bot):
        bar = ""
        for i in range(10):
            front_emoji = discord.utils.get(bot.emojis, id=Quest_Progress.Progress_Bar_MAPPING["front_full" if progress >= (i + 1) * 0.1 else "front_empty"]) or ":_:"
            mid_emoji = discord.utils.get(bot.emojis, id=Quest_Progress.Progress_Bar_MAPPING["mid_full" if progress >= (i + 1) * 0.1 else "mid_empty"]) or ":_:"
            bar += f"{front_emoji}{mid_emoji}" if i == 0 else f"{mid_emoji}"
        back_emoji = discord.utils.get(bot.emojis, id=Quest_Progress.Progress_Bar_MAPPING["back_full" if progress == 1 else "back_empty"]) or ":_:"
        bar += f"{back_emoji}"  # Back part emoji
        return bar
    
class QuestEmbed:
    @staticmethod
    async def create_quest_embed(
        quest: str,
        quest_id: int, 
        action: str, 
        method: str, 
        channel: discord.TextChannel, 
        times: int, 
        content: str
    ) -> discord.Embed:
        description = (
            f"**Action:** `{action}`\n"
            f"**Method:** `{method}`\n"
            f"**Channel:** {channel.mention}\n"
            f"**Times:** `{times}`\n"
            f"**Content:** `{content}`\n"
            f"**Quest:**\n```{action} {method}: '{content}' in {channel.name} {times}x```"
        )
        
        embed = discord.Embed(
            title=f"{quest} Quest #{quest_id}",
            description=description,
            color=None
        )
        embed.set_thumbnail(url=AnyaImages.new_mission_anya)
        
        return embed

class LogConstants:
    start_log_thumbnail = "https://example.com/start_log_thumbnail.png"
    footer_text = "Please commit your changes to the repository."
    footer_icon = "https://example.com/footer_icon.png"
    author_name = "Your Bot Name"
    author_icon = "https://example.com/author_icon.png"
    embed_color = None

class PingConstants:
    thumbnail_url = AnyaImages.ping_thumbnail
    image_url =  None
    footer_icon = None
    embed_color = None

    current_time = datetime.now().strftime("%I:%M:%S %p")
    system_info = {
        "Operating System": f"{platform.system()} {platform.release()}\n└── {platform.version()} ",
        "System Time": current_time,
        "Processor": platform.processor(),
        "Python Version": platform.python_version(),
        "System Version": current_time
    }
    language_info = {
        "Language": "Python",
        "Discord Library": f"+ {discord.__version__}\n└── discord.py"
    }
    @staticmethod
    def format_diff(value):
        cpu_threshold = 80
        python_version_threshold = "3.10.13"
        if isinstance(value, str) and value < python_version_threshold:
            return f"- {value}\n"
        elif isinstance(value, (int, float)) and value > cpu_threshold:
            return f"- {value}\n"
        else:
            return f"+ {value}\n"

# Classes
class Emojis:
    emoji_paths = {
        "cpu_emoji": "Emojis/cpu.png",
        "memory_emoji": "Emojis/memory.png",
        "python_emoji": "Emojis/python.png"
    }

    @staticmethod
    async def load(bot, ctx):
        print('Loading emojis from local files')
        emojis = {}
        for emoji_name, file_path in Emojis.emoji_paths.items():
            emoji = await Emojis.create_emoji(ctx, file_path, emoji_name)
            emoji_id = emoji.split(":")[-1][:-1]
            emoji_format = f"<:_:{emoji_id}>"
            emojis[emoji_id] = emoji_format  # Store emoji format with emoji ID as key
            print(emoji_format)

        formatted_results = ""
        for emoji_id in emojis.keys():
            formatted_results += f"<:_:{emoji_id}>"

        return formatted_results

    @staticmethod
    async def create_emoji(ctx, file_path, emoji_name):
        # Check if emoji.json file exists, create if not
        if not os.path.exists("Data"):
            os.makedirs("Data")
        if not os.path.exists("Data/emoji.json"):
            with open("Data/emoji.json", "w") as f:
                json.dump({}, f)  # Empty json file

        # Load emoji data from emoji.json
        with open("Data/emoji.json", "r") as f:
            emoji_data = json.load(f)

        # Check if emoji exists in the json
        if emoji_name in emoji_data and "emoji_id" in emoji_data[emoji_name]:
            # Emoji ID exists in json, return the emoji format
            emoji_id = emoji_data[emoji_name]["emoji_id"]
            return f"<:_:{emoji_id}>"

        # Check if emoji exists in the guild
        existing_emoji = discord.utils.get(ctx.guild.emojis, name=emoji_name)
        if existing_emoji:
            # Delete existing emoji from the guild
            await existing_emoji.delete()
            print(f"Deleted existing emoji: {existing_emoji.name}")

        # Create new emoji
        with open(file_path, "rb") as f:
            image = Image.open(f)
            img_byte_array = io.BytesIO()
            image.save(img_byte_array, format=image.format)
            img_byte_array.seek(0)
            print(f"Creating emoji from file: {file_path}")
            new_emoji = await ctx.guild.create_custom_emoji(name=emoji_name, image=img_byte_array.read())
            print(f"Emoji created successfully: {new_emoji.name}")

            # Update emoji data
            emoji_data[emoji_name] = {"emoji_id": new_emoji.id, "emoji_name": "_"}
            with open("Data/emoji.json", "w") as f:
                json.dump(emoji_data, f, indent=4)  # Save updated json with indentation

        # Return the emoji format
        return f"<:_:{new_emoji.id}>"

    
# Functions
def primary_color(image_path='Data/Images/bot_icon.png'):
    image = Image.open(image_path)
    # Resize the image to 1x1 pixel to get the dominant color
    resized_image = image.resize((1, 1))
    dominant_color = resized_image.getpixel((0, 0))
    return discord.Color.from_rgb(dominant_color[0], dominant_color[1], dominant_color[2])

# Async Functions
async def error_custom_embed(bot, ctx, e, title="Custom Error", thumbnail_url=AnyaImages.question_anya):
    error_embed = discord.Embed(
        description=f'```bash\n{e}```',
        color=discord.Color.red(),
        timestamp=datetime.now()
    )
    error_embed.set_author(name=f'{bot.user.display_name.title()} - {title}', icon_url=bot.user.avatar)
    line_number = traceback.extract_tb(e.__traceback__)[-1].lineno
    tb_frame = traceback.extract_tb(e.__traceback__)[-1]
    file_location = tb_frame.filename
    error_embed.add_field(
        name=" ",
        value=f"**Potential issue found:**\n- **File:** `{file_location}`\n- **Line:** `{line_number}`",
        inline=False
    )
    error_embed.set_footer(text='Error Found')
    error_embed.set_thumbnail(url=thumbnail_url)
    
    # Check if messageable is a Context (ctx) object or Interaction object and send message accordingly
    if isinstance(ctx, commands.Context):
        await ctx.reply(embed=error_embed)
    elif isinstance(ctx, discord.Interaction):
        await ctx.response.send_message(embed=error_embed)

    
 
