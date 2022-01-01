# Reinforcement Learning on DinoGame
Applying Reinforcement Learning to play "T-Rex Chrome Dino Game"

# 1. Features
## 1.1 Model Training
* training script: `training_pipeline/run_training.py`
* observation space: can select between image or coordinate inputs
* observation space: apply concatenation to allow the model to be able to look-back **X** steps before each predictions
* policy selection: `CnnPolicy` of image-inputs or `MlpPolicy` for coordinate-inputs
* save gameplay-videos while training
* training configs and metrics are stored in Weights & Biases

## 1.2 Model Evaluation
* inference script: `training_pipeline/model_evaluation.py`
* able to inference the model prediction results as a gameplay-video

# 2. Setups
## 2.1 Weights & Biases setup

# 3. Installation
## 3.1 Docker approach
## 3.2 Virtualenv approach
## 3.3 Main Dependencies

# 4. How to run
## 4.1 Run model training
## 4.2 Run model evaluation
