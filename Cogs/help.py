# Import necessary modules
import os
import traceback
import json

import cv2
import numpy as np
import aiohttp
import requests
import json
import os
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont, ImageSequence


from Imports.discord_imports import * 
from Data.const import primary_color, error_custom_embed, Help_Select_Embed_Mapping, Help_Embed_Mapping, banner_url
from Imports.log_imports import logger

class Select(discord.ui.Select):
    def __init__(self, cog_commands, bot, primary_color):
        options = [
            discord.SelectOption(label=cog_name.replace('_', ' '), value=cog_name, emoji=Help_Select_Embed_Mapping.emojis.get(cog_name.lower()))
            for cog_name in cog_commands.keys()
        ]
        if not options:
            options = [discord.SelectOption(label="No Categories Available", value="none")]
        super().__init__(placeholder="More Details...", max_values=1, min_values=1, options=options)
        self.cog_commands = cog_commands
        self.bot = bot
        self.page = 0  # Track the current page
        self.primary_color = primary_color
        self.command_mapping_file = 'Data/Help/command_map.json'

    def _ensure_file_exists(self):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.command_mapping_file), exist_ok=True)
        # Ensure the file exists
        if not os.path.exists(self.command_mapping_file):
            with open(self.command_mapping_file, 'w') as f:
                json.dump({}, f, indent=4)

    def _load_command_mapping(self):
        self._ensure_file_exists()
        with open(self.command_mapping_file, 'r') as f:
            return json.load(f)

    def _save_command_mapping(self, mapping):
        with open(self.command_mapping_file, 'w') as f:
            json.dump(mapping, f, indent=4)

    def _update_command_mapping(self):
        mapping = self._load_command_mapping()
        for cog_name in self.cog_commands.keys():
            if cog_name not in mapping:
                mapping[cog_name] = {}
            cog = self.bot.get_cog(cog_name)
            if cog:
                cog_commands = [cmd for cmd in cog.get_commands() if not cmd.hidden]
                for cmd in cog_commands:
                    if cmd.name not in mapping[cog_name]:
                        mapping[cog_name][cmd.name] = "Description to fill out"
        self._save_command_mapping(mapping)

   
    async def callback(self, interaction: discord.Interaction):
        try:
            self._update_command_mapping()

            cog_name = self.values[0]
            if cog_name == "none":
                await interaction.response.send_message("No categories available.", ephemeral=True)
                return

            cog_info = self.cog_commands.get(cog_name)
            color = self.primary_color
            emoji = Help_Select_Embed_Mapping.emojis.get(cog_name.lower())

            self.cog_embed1 = discord.Embed(
                title=f'Category - {emoji} {cog_name.replace("_", " ")}',
                description=f'{Help_Select_Embed_Mapping.embeds[cog_name.lower()]["description"] or ""}',
                color=color
            )
            self.cog_embed2 = discord.Embed(title=f'{emoji} {cog_name.replace("_", " ")}', description='', color=color)

            file = None
            if 'ai' in Help_Select_Embed_Mapping.embeds and cog_name.lower() == 'ai':
                file_path = 'Data/Images/Help_Thumbnails/ai.png'
                if os.path.exists(file_path):
                    file = discord.File(file_path, filename='thumbnail.png')
                    self.cog_embed2.set_thumbnail(url='attachment://thumbnail.png')
                else:
                    logger.error(f"Thumbnail file '{file_path}' not found.")
            else:
                self.cog_embed2.set_thumbnail(url=Help_Select_Embed_Mapping.embeds[cog_name.lower()]["thumbnail_url"])

            cog = self.bot.get_cog(cog_name)
            if cog:
                cog_commands = [cmd for cmd in cog.get_commands() if not cmd.hidden]
                if cog_commands:
                    command_mapping = self._load_command_mapping().get(cog_name, {})
                    for cmd in cog_commands:
                        cmd_args = [
                            f"[{param.name}]" if param.default is not param.empty else f"<{param.name}>"
                            for param in cmd.clean_params.values()
                        ]
                        args_str = " ".join(cmd_args)
                        command_info = f"`...{cmd.name}`  {command_mapping.get(cmd.name, 'No description available')}"
                        
                        self.cog_embed2.add_field(
                            name='',
                            value=command_info,
                            inline=False
                        )
                else:
                    logger.info(f"No visible commands found for cog: {cog_name}")
            else:
                logger.info(f"Cog not found: {cog_name}")

            if file:
                await interaction.response.edit_message(embed=self.cog_embed2, attachments=[file])
            else:
                await interaction.response.edit_message(embed=self.cog_embed2, attachments=[])


            logger.info("Message edited successfully.")
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(traceback_str)
            logger.debug(f"An error occurred: {traceback_str}")
            pass
        
        
class HelpMenu(discord.ui.View):
    def __init__(self, bot, primary_color, select_view, *, timeout=None):
        super().__init__(timeout=timeout)
        
        self.bot = bot
        self.primary_color = primary_color
        
        cog_commands = {}  # Dictionary to store commands for each cog
        self.add_item(select_view)
        # Send only the embed without the button


        
        
class ImageGenerator:
    def __init__(self, bot, avatar_url, background_url, font_path='Data/bubbles.ttf'):
        self.bot = bot
        self.avatar_url = avatar_url
        self.background_url = background_url
        self.font_path = font_path
        self.avatar = self.load_image_from_url(self.avatar_url)
        self.background = self.load_image_from_url(self.background_url)
        self.avatar_size = (100, 100)  # Adjust as needed
        self.command_mapping_file = 'Data/Help/command_map.json'

    def load_image_from_url(self, url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    def get_max_font_size(self, text, max_width, min_size=10, max_size=100):
        for size in range(max_size, min_size - 1, -1):
            font = ImageFont.truetype(self.font_path, size)
            text_bbox = ImageDraw.Draw(Image.new('RGBA', (1, 1))).textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            if text_width <= max_width:
                return size
        return min_size

    def _ensure_file_exists(self):
        os.makedirs(os.path.dirname(self.command_mapping_file), exist_ok=True)
        if not os.path.exists(self.command_mapping_file):
            with open(self.command_mapping_file, 'w') as f:
                json.dump({}, f, indent=4)

    def _load_command_mapping(self):
        self._ensure_file_exists()
        with open(self.command_mapping_file, 'r') as f:
            return json.load(f)

    def _save_command_mapping(self, mapping):
        with open(self.command_mapping_file, 'w') as f:
            json.dump(mapping, f, indent=4)

    def _update_command_mapping(self):
        mapping = self._load_command_mapping()
        for cog_name in self.cog_commands.keys():
            if cog_name not in mapping:
                mapping[cog_name] = {}
            cog = self.bot.get_cog(cog_name)
            if cog:
                cog_commands = [cmd for cmd in cog.get_commands() if not cmd.hidden]
                for cmd in cog_commands:
                    if cmd.name not in mapping[cog_name]:
                        mapping[cog_name][cmd.name] = "Description to fill out"
        self._save_command_mapping(mapping)

    def create_slideshow(self, output_path='slideshow.gif'):
        frames = []
        command_mapping = self._load_command_mapping()

        # Check if background is an animated GIF
        background_is_gif = self.background_url.lower().endswith('.gif')
        background_frames = []

        if background_is_gif:
            for frame in ImageSequence.Iterator(self.background):
                if frame.mode != 'RGBA':
                    frame = frame.convert('RGBA')
                background_frames.append(frame.copy())
        else:
            background_frames.append(self.background)

        # Create one frame per command with individual text
        for cog_name, commands in command_mapping.items():
            for cmd_name, cmd_desc in commands.items():
                for bg_frame in background_frames:
                    frame = bg_frame.copy()
                    
                    # Create a translucent black shader layer
                    shader_layer = Image.new('RGBA', frame.size, (0, 0, 0, 0))
                    shader_draw = ImageDraw.Draw(shader_layer)
                    shader_color = (0, 0, 0, 100)  # Translucent black
                    shader_draw.rectangle([(0, 0), (frame.width, frame.height)], fill=shader_color)
                    frame = Image.alpha_composite(frame, shader_layer)

                    # Position of the avatar (left of the text)
                    avatar_position = (50, (frame.height - self.avatar_size[1]) // 2)

                    # Get the maximum font size that fits the text
                    text_area_width = frame.width - self.avatar_size[0] - 70
                    text = f"...{cmd_name}"
                    font_size = self.get_max_font_size(text, text_area_width)
                    font = ImageFont.truetype(self.font_path, font_size)

                    # Calculate text size and position
                    draw = ImageDraw.Draw(frame)
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_height = text_bbox[3] - text_bbox[1]
                    text_position = (avatar_position[0] + self.avatar_size[0] + 20, (frame.height - text_height) // 2)

                    # Paste the avatar onto the background
                    mask = self.avatar.split()[3]
                    frame.paste(self.avatar, avatar_position, mask)

                    # Add styled text
                    text_color = (255, 255, 255)  # White color
                    draw.text(text_position, text, font=font, fill=text_color)

                    # Add a border around the text for better visibility
                    border_width = 2
                    border_color = (0, 0, 0)  # Black border
                    draw.text((text_position[0] - border_width, text_position[1] - border_width), text, font=font, fill=border_color)
                    draw.text((text_position[0] + border_width, text_position[1] - border_width), text, font=font, fill=border_color)
                    draw.text((text_position[0] - border_width, text_position[1] + border_width), text, font=font, fill=border_color)
                    draw.text((text_position[0] + border_width, text_position[1] + border_width), text, font=font, fill=border_color)

                    frames.append(frame.copy())  # Add the frame to the slideshow

        # Save the animated GIF
        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=1000)  # 1000ms per frame

        return output_path
    
        
class Help(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.cog_commands = {}
        self.command_mapping_file = 'Data/Help/command_map.json'

    def _ensure_file_exists(self):
        os.makedirs(os.path.dirname(self.command_mapping_file), exist_ok=True)
        if not os.path.exists(self.command_mapping_file):
            with open(self.command_mapping_file, 'w') as f:
                json.dump({}, f, indent=4)

    def _load_command_mapping(self):
        self._ensure_file_exists()
        with open(self.command_mapping_file, 'r') as f:
            return json.load(f)

    def _save_command_mapping(self, mapping):
        with open(self.command_mapping_file, 'w') as f:
            json.dump(mapping, f, indent=4)

    def _update_command_mapping(self):
        mapping = self._load_command_mapping()
        for cog_name in self.cog_commands.keys():
            if cog_name not in mapping:
                mapping[cog_name] = {}
            cog = self.bot.get_cog(cog_name)
            if cog:
                cog_commands = [cmd for cmd in cog.get_commands() if not cmd.hidden]
                for cmd in cog_commands:
                    if cmd.name not in mapping[cog_name]:
                        mapping[cog_name][cmd.name] = "Description to fill out"
        self._save_command_mapping(mapping)

    def format_cog_commands(self, cog_name, cog_commands, command_mapping):
        embed = discord.Embed(title=f"Commands for {cog_name}", color=primary_color_value)
        embed.description = ' '  # Add a placeholder description to ensure it's not empty

        for cmd_name in cog_commands:
            cmd = command_mapping.get(cmd_name)
            if cmd is None:
                continue  # Skip if command not found

            if not hasattr(cmd, 'clean_params'):
                continue  # Skip if cmd does not have clean_params

            cmd_args = [
                f"[{param.name}]" if param.default is not param.empty else f"<{param.name}>"
                for param in cmd.clean_params.values()
            ]
            args_str = " ".join(cmd_args)
            command_info = f"...{cmd.name} {args_str}"

            embed.add_field(name=cmd_name, value=f'```{command_info}```', inline=False)

        return embed

    @commands.command(hidden=True)
    async def help(self, ctx, command_name: str = None):
        try:
            cog_commands = {}
            command_mapping = self._load_command_mapping()

            bot_avatar_url = str(self.bot.user.avatar.with_size(128))
            async with aiohttp.ClientSession() as session:
                async with session.get(bot_avatar_url) as resp:
                    if resp.status != 200:
                        return await ctx.reply('Failed to get bot avatar.')
                    data = await resp.read()
            avatar_image = Image.open(BytesIO(data))
            temp_image_dir = 'Data/Images'
            temp_image_path = os.path.join(temp_image_dir, 'bot_icon.png')
            if not os.path.exists(temp_image_dir):
                os.makedirs(temp_image_dir)
            avatar_image.save(temp_image_path)
            primary_color_value = primary_color(temp_image_path)
        except Exception as e:
            logger.error(f"Error getting primary color: {e}")
            await ctx.reply(embed=await error_custom_embed(self.bot, ctx, e, title="Primary Color"))
            return

        if command_name is not None:
            command = self.bot.get_command(command_name)
            if command is not None:
                command_string = f"{command.qualified_name}  {command.signature.replace('[', '<').replace(']', '>').replace('=None', '')}"
                usage = f"{ctx.prefix}{command_string}"
                help_embed = discord.Embed(
                    title=command.qualified_name,
                    color=primary_color_value,
                    description=f"> **{command.help}**"
                )
                help_embed.add_field(name="Usage", value=f"```{usage}```", inline=True)
                await ctx.send(embed=help_embed)
            else:
                await ctx.send("Invalid command name. Please provide a valid command.")
        else:
            try:
                for cog_name, cog_object in self.bot.cogs.items():
                    if isinstance(cog_object, commands.Cog):
                        commands_in_cog = [cmd for cmd in cog_object.get_commands() if not cmd.hidden]
                        if commands_in_cog:
                            cog_commands[cog_name] = commands_in_cog

                self.cog_commands = cog_commands
                self._update_command_mapping()

                # Create the image slideshow
                image_generator = ImageGenerator(self.bot, avatar_url=bot_avatar_url, background_url='https://i.pinimg.com/originals/5a/35/1a/5a351aa5067e01fa2e00db8b4191c999.gif')
                image_file =  image_generator.create_slideshow()

                # Create the select view and help menu
                select_view = Select(self.cog_commands, self.bot, primary_color_value)
                help_menu = HelpMenu(self.bot, primary_color_value, select_view)

                await ctx.send(file=discord.File(image_file, 'help_slideshow.gif'), view=help_menu)
                await ctx.defer()
            except Exception as e:
                logger.error(f"Error sending HelpMenu: {e}")
                await ctx.reply(embed=await error_custom_embed(self.bot, ctx, e, title="Help"))
                return
            
            
def setup(bot):
    bot.add_cog(Help(bot))
