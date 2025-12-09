import os
import time
import random 
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTES:
# This has only been tested on windows while running in vscode
# when running this script, it assumes you are running this while the main folder is on the desktop

# WARNING ~ leaving this message here just in case, but from what I researched by running this 
# script we might be potentially legally viable to violating the Tos of the chosen website to 
# be scraped ('bing' in this case), copyright infringment of IPs, etc...
# For this reason it might be best to do some additional research on the ethical effects of image scrapping.

# !Code will run and a test version (old ver. chrome) of chrome will appear, do not touch the window 
# (you can slide it to the slide if needed while running), You can leave the window be or running in the 
# background as it performs its task! (reason we want to avoid touching the screen is avoid clicking the browser when the 
# script is trying to acess a website altering the scrape results)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#PACKAGES NEEDED
# type this in the teminal to download packages(vscode terminal if loading in vscode): 
# " pip install selenium requests "")

# prompts are modifyable in the "search_config" section

#_____________________________________________________________________________________________________________________

# --- Configuration ---
# IMPORTANT: Adjust these paths to match your system. Modify the following :C\Users\#### with your pc's user
CHROME_DRIVER_PATH = r'C:\Users\####\Desktop\Web_scraper\chrome-win64\chromedriver.exe'
CHROME_BINARY_PATH = r'C:\Users\####\Desktop\Web_scraper\chrome-win64\chrome.exe'
#IMPORTANT ~ Modify #### with your user
Save_folder =  r"C:\Users\####\Desktop\Web_scraper\data\train"

#_____________________________________________________________________________________________________________________

IMAGE_DOWNLOAD_TIMEOUT = 60 
#modifying the variable below is basically the maximum num of images that can be downloaded per run
MAX_IMAGES_PER_QUERY = 14000
PAUSE_MIN = 5
PAUSE_MAX = 10
BING_URL = "https://www.bing.com/images/search?q="

# --- Image Search Terms and Directory Structure ---
# modify the search prompts Here (can have as many prompts as you want, so in moderation: go wild!):
# the provided prompts were the prompts used to generate the initial images (50 prompts), on estimate 
# it was observed that every query generated/scraped ~100-130 images per prompt.
search_config = {
    
    #the following are examples of prompts that can be placed the the webscraper
    "Plushie_Cat": [

    "photorealistic portrait of a young woman smiling",
    "close-up photo of an elderly man with deep wrinkles",
    "studio headshot of a professional business executive",
    "detailed pencil sketch of a teenage girl with freckles",
    "oil painting of a serious-looking historical figure's face",
    "candid photo of a person laughing heartily outdoors",
    "low-key dramatic portrait of a mysterious figure in shadow",
    "high-resolution photo of a child's face, wide-eyed",
    "charcoal drawing of an expressive face, showing anger",
    "digital painting of a sci-fi character with glowing eyes",
    "vintage black and white photo of a 1920s flapper",
    "watercolour portrait of a person with colorful hair",
    "3D model render of an average-looking person's face",
    "close-up photo of a face with vibrant artistic makeup",
    "hyperrealistic drawing of a face with reflected light",
    "photo of a face in silhouette, backlit by the sun",
    "art nouveau style portrait of a serene young woman",
    "pixel art rendition of a well-known celebrity face",
    "comic book style illustration of a superhero mask off",
    "face of a person wearing traditional African jewelry",
    "a face showing intense surprise, mouth slightly open",
    "photorealistic face of a person with a skeptical look",
    "face reflected in a polished metal surface",
    "chalk pastel drawing of a realistic face on textured paper",
    "face portrait with abstract geometric shapes overlay",
    "side profile of a face with dramatic shadow play",
    "face of a tired person leaning against a wall",
    "realistic photo of a person making a humorous grimace",
    "portrait of a woman with braided hair, natural setting",
    "face of a person wearing a colorful knitted scarf"
    ],

    "AI_Cat": [
    ]
}

# --- Core Scraper Functionality ---

def initialize_driver():
    """Initializes and returns the Chrome WebDriver with anti-detection settings."""
    print("--- Entering driver initialization ---")
    try:
        chrome_options = Options()
        chrome_options.binary_location = CHROME_BINARY_PATH
        
        # --- ANTI-DETECTION SETTINGS ---
        # mainly here due to bot detection in websites that make it hard to web scrape
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        # -----------------------------
        
        # chrome_options.add_argument("--headless") # Uncomment to run without visible browser window
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        service = Service(executable_path=CHROME_DRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Set a global implicit wait for finding elements
        driver.implicitly_wait(IMAGE_DOWNLOAD_TIMEOUT)
        
        print(f"Driver initialized successfully using binary: {CHROME_BINARY_PATH}")
        # Execute script to spoof navigator.webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        print("Applied anti-detection scripts.")
        return driver
    except Exception as e:
        print("\n!!! DRIVER INITIALIZATION FAILED !!!")
        if "session not created: This version of ChromeDriver only supports Chrome version" in str(e):
            print("FATAL ERROR: Version Mismatch. ChromeDriver does not match Chrome for Testing version.")
        elif "not found" in str(e) or "No such file" in str(e):
            print(f"FATAL ERROR: ChromeDriver not found at '{CHROME_DRIVER_PATH}' or Chrome binary missing.")
        else:
            print(f"An unexpected error occurred: {e}")
        return None

def scrape_bing_images(driver, search_term, download_dir, existing_count):
    """Searches Bing, scrolls to find image URLs, and downloads them."""
    url = f"{BING_URL}{search_term.replace(' ', '+')}"
    print(f"\nSearching Bing for: '{search_term}'...")

    try:
        driver.get(url)
        
        # Waits for the main image grid element to be visible. *Critical*
        # This uses the longer IMAGE_DOWNLOAD_TIMEOUT (60s)
        WebDriverWait(driver, IMAGE_DOWNLOAD_TIMEOUT).until( 
            EC.presence_of_element_located((By.CSS_SELECTOR, "img.mimg"))
        )

        print("Initial search results loaded successfully.")
        
    except Exception as e:
        print(f"TIMEOUT: Failed to load initial search results within {IMAGE_DOWNLOAD_TIMEOUT} seconds. Skipping this query.")
        # print(f"Error details: {e}") # can be uncommented for debugging
        return 0

    image_urls = set()
    scroll_attempts = 0
    max_scrolls = 25 # Can Increase scrolling to potentially find more images
    
    # Scroll the page to load more images
    while len(image_urls) < MAX_IMAGES_PER_QUERY and scroll_attempts < max_scrolls:
        # Get all image tags
        current_images = driver.find_elements(By.CSS_SELECTOR, "img.mimg")
        
        # Extract the source URL from the image tag
        for img in current_images:
            #looking for the 'm' or 'src' attribute
            src = img.get_attribute('m') or img.get_attribute('src')
            if src and src.startswith('http'):
                image_urls.add(src)

        # Scroll down to trigger loading of more images
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5) # Short pause after scroll
        scroll_attempts += 1
        
        if len(image_urls) >= MAX_IMAGES_PER_QUERY:
            break

    print(f"Found {len(image_urls)} potential image URLs for '{search_term}'. Starting download...")

    # Download Images
    new_downloads = 0
    
    # Check if a download is still needed (in case the query reached the max limit earlier)
    if existing_count + new_downloads >= MAX_IMAGES_PER_QUERY:
        print("Max images reached for this class before download attempt.")
        return 0

    for img_url in image_urls:
        if new_downloads >= MAX_IMAGES_PER_QUERY:
            break
            
        # Create unique filename based on current total download count
        filename = f"{existing_count + new_downloads:04d}.jpg"
        filepath = os.path.join(download_dir, filename)

        try:
            # Use requests for a faster, independent download
            response = requests.get(img_url, timeout=15) # Using 15s timeout for download
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                new_downloads += 1
                # Optional: print(f"Downloaded: {filename}")
        except Exception as e:
            # Silently skip failed downloads (timeouts, connection errors, etc.)
            pass 

    print(f"Successfully downloaded {new_downloads} new images for '{search_term}'.")
    return new_downloads

def main():
    """Main execution flow for the image scraper."""
    
    print("--- Script execution started ---")
    
    total_new_images = 0
    
    # 1. Initialize WebDriver
    driver = initialize_driver()
    if driver is None:
        print("Cannot proceed without a working WebDriver.")
        return

    # 2. Scrape Images for Each Class
    for class_name, search_terms in search_config.items():
        base_dir = Save_folder
        download_dir = os.path.join(base_dir, class_name)
        os.makedirs(download_dir, exist_ok=True)

        print(f"\n{'='*55}")
        print(f"STARTING SCRAPE FOR CLASS: {download_dir}")
        print(f"{'='*55}")

        # Determine existing count to start numbering new files
        existing_files = [f for f in os.listdir(download_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        existing_count = len(existing_files)
        print(f"Starting download index in {download_dir} from {existing_count}...")
        
        class_new_images = 0

        for term in search_terms:
            # Random pause to avoid being blocked
            pause_time = random.uniform(PAUSE_MIN, PAUSE_MAX) 
            print(f"Pausing for {pause_time:.1f} seconds before starting new search: '{term}'...")
            time.sleep(pause_time)

            # Scrape and download
            downloaded = scrape_bing_images(driver, term, download_dir, existing_count + class_new_images)
            class_new_images += downloaded
        
        total_new_images += class_new_images
        print(f"\n{'-'*30}")
        print(f"Total new images saved to {download_dir}: {class_new_images}")
        print(f"{'-'*30}")

    # 3. Cleanup and Summary
    driver.quit()
    print(f"\n{'='*50} GRAND SCRAPING COMPLETE {'='*50}")
    print(f"Total new images saved: {total_new_images}")

if __name__ == "__main__":
    main()