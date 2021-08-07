# The African Eagles

## Table of Contents

   * [The African Eagles](#the-african-eagles)
   * [Table of Contents](#table-of-contents)
      * [About the team](#about-the-team)
      * [About our project "TRIA"](#about-our-project-tria)
      * [How to run](#how-to-run)
         * [Step 1: Train the ML model](#step-1-train-the-ml-model)
         * [Step 2: Run project](#step-2-run-project)
         * [Step 3: Predict](#step-3-predict)
         * [Step 4: Generate summary data](#step-4-generate-summary-data)
      * [Watch the demo](#watch-the-demo)
      * [License](#license)

## About the team

We are a team of 4 members:  

| Front End Developer | Team Lead | AI Engineer | AI Engineer |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|<img src="https://avatars.githubusercontent.com/u/11523791?v=4" width="100px" height="100px"> |  <img src="https://avatars.githubusercontent.com/u/25987558?v=4" width="100px" height="100px"> | <img src="https://avatars2.githubusercontent.com/u/27445092?s=460&u=349cffccfccda38293e4aab20868a77b60079274&v=4" width="100px" height="100px"> | <img src="https://avatars.githubusercontent.com/u/53430747?v=4" width="100px" height="100px">|
|[Abderrahim SOUBAI-ELIDRISI](https://github.com/AbderrahimSoubaiElidrissi)| [Fatima-Ezzahra](https://github.com/Fatiima-Ezzahra) | [Sara EL-ATEIF](https://github.com/elateifsara)| [Ikechukwu Nigel Ogbuchi](https://github.com/Ogbuchi-Ikechukwu) |

## About our project "TRIA"

**Purpose :**  
Leveraging Computer Vision Techniques for Image-based Food Quality Evaluation.

**Solution :**  
Food quality evaluation system using the OAK-D to assist food suppliers and consumers on appropriate food and waste management.

## How to run

### Requirements

- Python 3.8
- OpenVino 
- Depthai library
- OAK-D device from OpenCV
- Python IDE (used PyCharm)

For all needed libraries please install from requirements.txt by typing in command line :
`pip install requirements.txt` 

### Step 1: Train the ML model

Navigate to the model training folder and run one of the notebooks provided in Colab. This will generate a yaml and bin file that you need to turn into blob by using the provided OpenVino platform as described in the notebooks. 

### Step 2: Run project

Create a virtual environment for your project and install necessary libraries. Then place one of the py files available in oakd files folder according to the model you trained. Finally, link the OAK-D device to your computer and run the py file.

### Step 3: Predict

This project was trained to predict rotten and fresh fruits, so you can use some real world fruits to keep track of their quality over time.

### Step 4: Generate summary data

At the end of the day, you can run the `post_processing.py` file to generate summary data to keep track of how many fruits were rotten or fresh.

## Watch the demo

[![TRIA Demo](http://img.youtube.com/vi/3IE6bdrxjSc/0.jpg)](https://youtu.be/3IE6bdrxjSc "TRIA Demo")

## License

See the [LICENSE](https://github.com/The-African-Eagles/back-end/blob/main/LICENSE) file for license rights and limitations (MIT).
