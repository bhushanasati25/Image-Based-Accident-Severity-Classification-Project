# Multiclass Object Classification in Autonomous Driving

## Table of Contents
1. [Project Overview](#project-overview)
2. [Team Members](#team-members)
3. [Project Description](#project-description)
4. [Dataset Links](#dataset-links)
5. [Technical Details](#technical-details)
6. [Installation & Setup](#installation--setup)
7. [Usage Guide](#usage-guide)
8. [Results & Performance](#results--performance)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [License & Contact](#license--contact)

## Project Overview

### Quick Start
```bash
git clone https://github.com/bhushanasati25/Multiclass-Object-Classification-in-Autonomous-Driving.git
cd Multiclass-Object-Classification-in-Autonomous-Driving
pip install -r requirements.txt
```

### Project Goals
- Develop a robust object classification system for autonomous driving
- Implement and compare 8 different classification models
- Create a deployable solution with real-time inference capabilities
- Provide comprehensive model comparison and analysis

### Classification Categories
1. **Multi-class Classification** (DenseNet-121 Specialized)**
   - Human (Pedestrian, Person_sitting, Cyclist)
   - Vehicle (Car, Truck, Van, Tram)

## Team Members

### Project Team
1. Bhushan Asati
   - Role: Data Scientist
   - Models: DenseNet-121, MobileNetV2 
   - Contributions: Data preprocessing, Feature Engineering, Model training, Model Optimiztion, Deployment

2. Rujuta Dabke
   - Role: Data Scientist
   - Models: EfficientNet, Vision Transformer
   - Contributions: Feature engineering, Visualization

3. Suyash Madhavi
   - Role: Data Scientist
   - Models: Inception-v3, ResNet-50
   - Contributions: Model optimization, Performance analysis

4. Anirudha Sharma
   - Role: Data Scientist
   - Models: XGBoost, Random Forest
   - Contributions: Traditional ML implementation, Evaluation metrics

## Project Description

### Project Scope
- Implementation of 8 different classification models
- Real-time inference capabilities
- Web interface for demonstration
- Comprehensive performance analysis
- Model comparison framework

### Expected Outputs
1. **Technical Deliverables**
   - Trained model weights
   - API for inference
   - Web interface
   - Performance metrics

2. **Documentation**
   - Technical documentation
   - API documentation
   - User guides
   - Performance reports

### Innovation Points
- Multi-model comparison framework
- Specialized DenseNet-121 implementation
- Hybrid classification approach
- Comprehensive evaluation metrics

## Dataset Links

### Kaggle_dataset: https://www.cvlibs.net/datasets/kitti/

### KITTI_dataset: https://www.kaggle.com/datasets/garymk/kitti-3d-object-detection-dataset

## Technical Details

### Project Structure
```
Multiclass-Object-Classification-KITTI/
├── data/
│   ├── raw/               
│   ├── processed/        
│   └── samples/           
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_model_training.ipynb
│   └── 3_model_evaluation.ipynb
├── models/
│   ├── resnet50_model.h5
│   ├── efficientnetb0_model.h5
│   └── ...
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── streamlit_app/
│   └── app.py
└── docker/
    └── Dockerfile
```

### Models Implemented

1. **ResNet-50**
   - Architecture: Deep residual network
   - Features: Skip connections, Batch normalization
   - Performance: 89% accuracy

2. **EfficientNet**
   - Architecture: Compound scaling
   - Features: Balanced scaling, Optimized architecture
   - Performance: 91% accuracy

3. **MobileNetV2**
   - Architecture: Lightweight CNN
   - Features: Inverted residuals, Linear bottlenecks
   - Performance: 87% accuracy

4. **Vision Transformer (ViT)**
   - Architecture: Transformer-based
   - Features: Attention mechanisms, Patch-based processing
   - Performance: 90% accuracy

5. **Inception-v3**
   - Architecture: Multi-scale processing
   - Features: Factorized convolutions, Auxiliary classifiers
   - Performance: 88% accuracy

6. **DenseNet-121**
   - Architecture: Dense connectivity
   - Features: Fine-tuned for detailed classification
   - Performance: 92% accuracy

7. **XGBoost**
   - Type: Gradient boosting
   - Features: Feature importance, Handles imbalanced data
   - Performance: 85% accuracy

8. **Random Forest**
   - Type: Ensemble learning
   - Features: Feature selection, Parallel processing
   - Performance: 83% accuracy

### Technical Stack
- Python 3.8+
- PyTorch
- TensorFlow
- scikit-learn
- XGBoost
- OpenCV
- Streamlit

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU
- Git
- Docker (optional)

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt

# Download dataset
python scripts/preprocess.py --download
```

### Dataset Setup
```bash
# Process dataset
python scripts/preprocess.py --process
```

## Usage Guide

### Training Models
```bash
# Train specific model
python scripts/train.py --model resnet50 --epochs 100

# Train all models
python scripts/train.py --all
```

### Evaluation
```bash
# Evaluate model
python scripts/evaluate.py --model resnet50
```

### Web Interface
```bash
cd streamlit_app
streamlit run app.py
```

## Results & Performance

### Model Comparison

| Model          | Accuracy | F1-Score | Training Time | GPU Memory |
|----------------|----------|----------|---------------|------------|
| ResNet-50      | 0.89     | 0.88     | 4h 30m       | 8GB         |
| EfficientNet   | 0.91     | 0.90     | 5h 15m       | 6GB         |
| MobileNetV2    | 0.87     | 0.86     | 3h 45m       | 4GB         |
| ViT            | 0.90     | 0.89     | 6h 30m       | 12GB        |
| Inception-v3   | 0.88     | 0.87     | 5h 00m       | 8GB         |
| DenseNet-121   | 0.98     | 0.96     | 7h 45m       | 18GB        |
| XGBoost        | 0.85     | 0.84     | 1h 30m       | CPU         |
| Random Forest  | 0.83     | 0.82     | 45m          | CPU         |

### Key Findings
1. DenseNet-121 showed best performance for detailed classification
2. Deep learning models consistently outperformed traditional ML approaches
3. MobileNetV2 provided best speed-accuracy trade-off
4. Vision Transformer showed promising results but required more training data

## Future Work
1. Implement ensemble methods
2. Add real-time video processing
3. Optimize for edge deployment
4. Expand dataset with synthetic data
5. Implement cross-validation

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License & Contact

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact Information
- Bhushan Asati : [basati@stevens.edu]
- Rujuta Dabke : [rdabke@stevens.edu]
- Suyash Madhavi: [smadhavi1@stevens.edu]
- Anirudha Sharma : [asharma16@stevens.edu]

### Repository
Project Link: [https://github.com/bhushanasati25/Multiclass-Object-Classification-in-Autonomous-Driving.git](https://github.com/bhushanasati25/Multiclass-Object-Classification-in-Autonomous-Driving.git)
