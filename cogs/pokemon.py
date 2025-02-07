# Standard Libraries
import os
import sqlite3
import asyncio
import requests
import numpy as np
import pandas as pd
from io import BytesIO

from urllib.parse import unquote
from concurrent.futures import ProcessPoolExecutor


import cv2
import numpy as np
import motor.motor_asyncio

from Imports.discord_imports import *




class Pokemon_Data:
    def __init__(self):
        self.DB_NAME = "Pokemon"
        self.COLLECTION_NAME = "Hunters"

        mongo_url = os.getenv("MONGO_URI")
        if not mongo_url:
            raise ValueError("No MONGO_URI found in environment variables")
        self.mongoConnect = motor.motor_asyncio.AsyncIOMotorClient(mongo_url)
        self.db = self.mongoConnect[self.DB_NAME]
        self.hunters_collection = self.db[self.COLLECTION_NAME]

        self.pokemon_df = pd.read_csv("Data/commands/pokemon/pokemon_description.csv")

    async def check_pokemon_exists(self, pokemon_name):
        """Check if a Pokémon exists in the database."""
        return not self.pokemon_df[
            self.pokemon_df["slug"].str.lower() == pokemon_name.lower()
        ].empty

    async def get_user_pokemon(self, user_id):
        """Get a user's list of Pokémon."""
        user_data = await self.hunters_collection.find_one({"user_id": user_id})
        if user_data:
            return user_data["pokemon_list"]
        return []

    async def add_pokemon_to_user(self, user_id, pokemon_name):
        """Add a Pokémon to a user's list."""
        user_pokemon = await self.get_user_pokemon(user_id)
        user_pokemon.append(pokemon_name)
        await self.hunters_collection.update_one(
            {"user_id": user_id}, {"$set": {"pokemon_list": user_pokemon}}, upsert=True
        )

    async def remove_pokemon_from_user(self, user_id, pokemon_name):
        """Remove a Pokémon from a user's list."""
        user_pokemon = await self.get_user_pokemon(user_id)
        user_pokemon = [p for p in user_pokemon if p.lower() != pokemon_name.lower()]
        await self.hunters_collection.update_one(
            {"user_id": user_id}, {"$set": {"pokemon_list": user_pokemon}}
        )

    async def get_hunters_for_pokemon(self, pokemon_name):
        """Get all hunters for a specific Pokémon."""
        hunters = await self.hunters_collection.find(
            {"pokemon_list": {"$in": [pokemon_name]}}
        ).to_list(None)
        
        return [hunter["user_id"] for hunter in hunters]


class Pokemon_Detection:
    def __init__(self, dataset_path="Data/commands/pokemon/pokemon_images", db_path="Data/commands/pokemon/pokemon_images.db"):
        self.dataset_path = dataset_path
        self.db_path = db_path
        self.create_db()
        self.dataset_images = self.load_images_from_db()
        if not self.dataset_images:
            print("Database is empty! Saving images to DB...")
            self.save_images_to_db()
            self.dataset_images = self.load_images_from_db()
        self.dataset_features = self.extract_features_parallel()

    def create_db(self):
        """Create a SQLite database to store images."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                          id INTEGER PRIMARY KEY,
                          filename TEXT,
                          image BLOB)''')
        conn.commit()
        conn.close()

    def load_images_from_db(self):
        """Load all images stored in the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filename, image FROM images")
        images = {}
        for filename, image_data in cursor.fetchall():
            img_array = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                images[filename] = img
        conn.close()
        return images

    def save_images_to_db(self):
        """Save images from the dataset directory to the SQLite database."""
        image_files = [file for file in os.listdir(self.dataset_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for file in image_files:
            img_path = os.path.join(self.dataset_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            _, img_encoded = cv2.imencode('.jpg', img)
            img_bytes = img_encoded.tobytes()
            cursor.execute("INSERT INTO images (filename, image) VALUES (?, ?)", (file, img_bytes))
        
        conn.commit()
        conn.close()

    @staticmethod
    def process_image_features(image_data):
        """Extract ORB features for multiprocessing."""
        name, img = image_data
        orb = cv2.ORB_create(nfeatures=200)  # Reduce the number of features for speed
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_img, None)

        # Flip image and compute features again
        flipped_img = cv2.flip(img, 1)
        flipped_gray = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2GRAY)
        keypoints_flip, descriptors_flip = orb.detectAndCompute(flipped_gray, None)

        return name, descriptors, f"flipped_{name}", descriptors_flip

    def extract_features_parallel(self):
        """Extract ORB features in parallel for better speed."""
        features = {}
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.process_image_features, self.dataset_images.items()))

        for name, descriptors, flipped_name, flipped_descriptors in results:
            if descriptors is not None:
                features[name] = descriptors
            if flipped_descriptors is not None:
                features[flipped_name] = flipped_descriptors
        return features

    @staticmethod
    def filter_redundant_descriptors(descriptors, threshold=5.0):
        """Remove redundant descriptors based on similarity."""
        if descriptors is None:
            return None
        filtered = []
        for desc in descriptors:
            if all(np.linalg.norm(desc - d) > threshold for d in filtered):
                filtered.append(desc)
        return np.array(filtered, dtype=np.uint8) if filtered else None

    def detect_and_match(self, query_img):
        """Detect features in the query image and match against the dataset in parallel."""
        orb = cv2.ORB_create(nfeatures=200)  # Reduce the number of features for speed
        gray_query = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        _, query_descriptors = orb.detectAndCompute(gray_query, None)

        if query_descriptors is None:
            return None, None

        # Filter redundant descriptors
        query_descriptors = self.filter_redundant_descriptors(query_descriptors)

        dataset_list = [(query_descriptors, name, desc) for name, desc in self.dataset_features.items()]
        
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.process_matching, dataset_list))

        best_match, best_score = min(results, key=lambda x: x[1], default=(None, float('inf')))


        return best_match, round(best_score, 2)

    @staticmethod
    def process_matching(data):
        """Match query descriptors to dataset descriptors."""
        query_descriptors, dataset_name, dataset_descriptors = data
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(query_descriptors, dataset_descriptors)
        if not matches:
            return dataset_name, float('inf')
        score = np.mean([match.distance for match in matches])
        return dataset_name, score


class Detection_Listener(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.predictor = Pokemon_Detection() 
        self.data_handler = Pokemon_Data()  
        self.author_id = 716390085896962058  # poketwo
        self.phrase = "Hunters:"
        self.detect_bot_id = [
            854233015475109888,
            874910942490677270,
        ]  

    @commands.command()
    async def predict(self, ctx, *, arg=None):
        image_url = None

        if arg:
            image_url = arg
        elif ctx.message.attachments:
            image_url = ctx.message.attachments[0].url
        elif ctx.message.reference:
            reference_message = await ctx.channel.fetch_message(
                ctx.message.reference.message_id
            )
            if reference_message.attachments:
                image_url = reference_message.attachments[0].url
            elif reference_message.embeds:
                embed = reference_message.embeds[0]
                if embed.image:
                    image_url = embed.image.url

        if image_url is None:
            await ctx.send("No image URL found.")
            return

        # Fetch the image
        try:
            response = requests.get(image_url)
            image_data = BytesIO(response.content)
            image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            await ctx.send(f"Error fetching image: {str(e)}")
            return

        # Perform prediction using the Pokemon_Detection class
        pokemon_name, score = await self.predictor.detect_and_match(image)

        if pokemon_name is None:
            await ctx.send("No matching Pokémon found.")
        else:
            await ctx.send(f"Predicted Pokémon: {pokemon_name} with a score of {score}")
            
    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.id == self.author_id and message.embeds:
            embed = message.embeds[0]
            if embed.description and "Guess the pokémon" in embed.description:
                image_url = embed.image.url

                bot_response = await self.wait_for_bot_response(message.channel)

                if not bot_response:
                    pokemon_name, score = await self.predictor.detect_and_match(image_url)
                    hunters = await self.data_handler.get_hunters_for_pokemon(pokemon_name)

                    if hunters:
                        hunter_mentions = []
                        for hunter_id in hunters:
                            member = message.guild.get_member(hunter_id)
                            if member is None:
                                await self.data_handler.remove_pokemon_from_user(hunter_id, pokemon_name)
                            else:
                                hunter_mentions.append(f"<@{hunter_id}>")

                        if hunter_mentions:
                            ping_message = f"{pokemon_name}: {score}\n{self.phrase} {' '.join(hunter_mentions)}"
                            await message.channel.send(ping_message, reference=message)
                        else:
                            await message.channel.send(f"{pokemon_name}: {score}", reference=message)
                    else:
                        await message.channel.send(f"{pokemon_name}: {score}", reference=message)


    async def wait_for_bot_response(self, channel):
        
        def check(msg):
            return msg.author.id in self.detect_bot_id and msg.channel == channel

        try:
            
            msg = await self.bot.wait_for(
                "message", timeout=self.wait_time, check=check
            )
            return msg
        except asyncio.TimeoutError:
            
            return None
        


def setup(bot):
    bot.add_cog(Detection_Listener(bot))