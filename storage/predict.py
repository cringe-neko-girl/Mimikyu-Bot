import os
import cv2 as cv
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time

class PokemonPredictor:
    def __init__(self, dataset_folder='pokemon_images', output_folder='testing_output', dataset_file="_dataset.npy"):
        self.orb = cv.ORB_create(nfeatures=175)
        self.flann = cv.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=9, multi_probe_level=1), 
                                          dict(checks=2))
        self.executor = ThreadPoolExecutor(max_workers=25)
        self.cache = {}
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.dataset_file = dataset_file
        asyncio.run(self.load_dataset(dataset_folder))

    async def load_dataset(self, dataset_folder):
        if os.path.exists(self.dataset_file):
            await self.load_from_npy(self.dataset_file)
        else:
            await self.load_from_images(dataset_folder)

    async def load_from_npy(self, dataset_file):
        data = np.load(dataset_file, allow_pickle=True).item()
        for filename, (descriptors, bounding_box) in data.items():
            self.cache[filename] = (descriptors, bounding_box)

    async def load_from_images(self, dataset_folder):
        tasks = [self.process_image(os.path.join(dataset_folder, filename), filename) 
                 for filename in os.listdir(dataset_folder) 
                 if filename.endswith('.png')]
        await asyncio.gather(*tasks)
        np.save(self.dataset_file, self.cache)

    async def process_image(self, path, filename):
        img = await self.executor.submit(cv.imread, path)
        if img is not None:
            compressed_img = self.compress_image(img)  # Compress the original image
            keypoints, descriptors = await self.executor.submit(self.orb.detectAndCompute, 
                                                               cv.cvtColor(compressed_img, cv.COLOR_BGR2GRAY), None)
            if descriptors is not None:
                bounding_box = self.calculate_bounding_box(keypoints)
                self.cache[filename] = (descriptors.astype(np.uint8), bounding_box)
                await self.executor.submit(self.save_image, compressed_img, filename)

    def compress_image(self, img, quality=90):
        """Compress an image to reduce file size."""
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv.imencode('.jpg', img, encode_param)
        if result:
            return cv.imdecode(encimg, 1)
        return img

    async def save_image(self, img, filename):
        """Save the compressed image to the output folder."""
        output_path = os.path.join(self.output_folder, filename)
        cv.imwrite(output_path, img)

    def calculate_bounding_box(self, keypoints):
        points = np.array([kp.pt for kp in keypoints], dtype=np.int32)
        return cv.boundingRect(points)

    async def compute_roi_density(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        density = cv.countNonZero(gray) / (gray.shape[0] * gray.shape[1])
        return density

    async def match(self, img, desB):
     roi_density = await self.compute_roi_density(img)
     if roi_density < 0.01:
        return None
    
     futures = [self.executor.submit(self.flann.knnMatch, desB, desA, k=2) 
               for _, (desA, _) in self.cache.items()]
     results = [future.result() for future in futures]
     best_match = self.process_knn_results(results)

     return best_match

    def process_knn_results(self, results):
     best_match = None
     max_accuracy = 0

     def evaluate_matches(filename, matches):
        # Calculate good matches
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        accuracy = len(good_matches) / len(matches) * 100 if matches else 0
        # Return the filename and accuracy for the best match
        return (filename, accuracy)

     with ThreadPoolExecutor() as executor:
        # Create a list of tasks for evaluating matches
        futures = {
            executor.submit(evaluate_matches, filename, matches): filename
            for filename, matches in zip(self.cache.keys(), results)
        }

        for future in as_completed(futures):
            try:
                filename, accuracy = future.result()
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_match = (filename, accuracy)
            except Exception as e:
                print(f"Error processing matches: {e}")

     return best_match
 
 
 
 
 
 
 
 
 
 
 
    async def predict_pokemon(self, img):
        start_time = time.time()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kpB, desB = self.orb.detectAndCompute(gray_img, None)
        best_match = await self.match(img, desB)

        if best_match:
            predicted_pokemon, accuracy = best_match
            predicted_name = predicted_pokemon.replace(".png", "")
            elapsed_time = time.time() - start_time
            return f"{predicted_name.title()}: {round(accuracy, 2)}%", elapsed_time
        
        return "No match found", time.time() - start_time

    def load_image_from_url(self, url):
        try:
            img = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
            return cv.imdecode(img, cv.IMREAD_COLOR)
        except requests.RequestException as e:
            print(f"Error fetching image from URL: {e}")
            return None

    def fluster(self, results):
        """Handles multiple predictions and sorts them based on accuracy."""
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)  # Sort by accuracy
        return sorted_results

if __name__ == "__main__":
    predictor = PokemonPredictor()

    while True:
        image_url = input("Enter the URL of the image to scan, type 'quit' to exit: ")
        if image_url.lower() == 'quit':
            break
        img = predictor.load_image_from_url(image_url)
        if img is not None:
            prediction, elapsed_time = asyncio.run(predictor.predict_pokemon(img))
            print(prediction)
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        else:
            print("Failed to load the image.")
