# ME369P-Final-Project-Public
1.The purpose of this project is to differentiate between an AI cat, a real cat, or a plushie cat. This is accomplished by training 3 models: a CNN, ResNet, and a Swin Transformer. The Swin Transformer performed the best. The models can be downloaded from the following links, and the data used to train these models. 
2. After an image is classified, the end effector of a robot is moved to a corresponding x,y coordinate based on its classification and then to a corresponding Z coordinate based on the model's confidence value. 

## model link 
https://drive.google.com/file/d/1LmHvKBR9CANGtaBeEO-KHJPdY2GyRZa_/view?usp=sharing
## data link 
https://drive.google.com/drive/folders/1-UtQ0X5nTJHMcUOAaTisVJr9Kmh7YTsy?usp=sharing
## quick guide 
The robot.py file takes in an input of a model that the user desires to use, and an image that the user wishes to evaluate.

## Data Directory Structure
The training script expects a specific folder structure within a local data/ directory. The dataset is typically split into 90% training data and 10% validation data. Ensure each subfolder is named exactly as the categories used in the model.

('This is an example of the general folder structure of the project, to use each model trainer')

```bash
ME369P-Final-Project/
├── Model_Training_cnn.py
└── data/
    ├── train/                    # ~ 90% Training Split
    │   ├── AI_Cat/               
    │   ├── Plushie_Cat/          
    │   └── Real_Cat/             
    └── val/                      # ~ 10% Validation Split
        ├── AI_Cat/
        ├── Plushie_Cat/
        └── Real_Cat/
```
## Running the Web Scraper
link to web scraper https://drive.google.com/file/d/1TiMitOJcaWLfoqK-pI29evQQuKas9FmE/view?usp=sharing

For privacy reasons and compliance with security practices, the full web scraper source code is not hosted directly on this GitHub repository.

GitHub’s policies on personally identifiable information (PII) and automated extraction tools can sometimes restrict the hosting of specific scraping configurations that might inadvertently expose sensitive endpoints or identifiers. Consequently, the scraper and its associated logic are available in the external data folder linked below.

1. Head to the data folder link to access the webscraper, locate "Web_scraper.zip"; download the file and extract the folder
2. Place the Web_scaper folder on your Desktop (all teh necessary files are alread pre-installed so only need to place folder)
3. Running the file on your IDE, locate the "configuration section"
4. In the configuration section there should be a baseline path in the file, for every "####" replace with your computers username.
6. Move on the "search_config" section, there should be 2 main categories available "Plushie_Cat" and "AI_Cat"
7. In between the [], insert your query/search for the search you are interested image search you are interested to scrape. "You can make multiple searches within the same session"
8. Run the script

## Running the CNN model (Model_Training_cnn.py)
Ensure you are using "python 3.10.11" for this model, as some packages do not suport python beyond this version:
~ The model was trained using VScode
1. Download the "Model_Training_cnn.py" file, load an IDE with the file loaded; create a virtual enviorment using the following command:
```bash
python -m venv cat_classifier_venv
```
2. To activate you current enviorment use the following commands dedpending if you are using git Bash or powershell (VScode_term/terminal)
```bash
PowerShell: .\cat_classifier_venv\Scripts\Activate.ps1
Bash: source cat_classifier_venv/bin/activate
```
4. To run the CNN model, run the following command in your terminal to install the necessary packages to run the model: 
```bash
pip install torch torchvision numpy matplotlib pillow
```
5. Run the model within the enviorment:
```bash
python Model_Training_cnn.py
```

## Running the Swin  model (swinModel.py)
works on Python 3.13
1. Download the "swinModel.py" file.
2. Verify packages are installed or install by running 3. pip install torch torchvision timm tqdm
3. Run Python SwinModel.py from the command line, as on certain systems, multi-processors will break if run from IDE.
4. If it fails, change subpro = 1 on line 70



