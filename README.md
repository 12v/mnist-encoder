# Image transformer

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

1. Install requirements from `requirements.txt`:
   ```bash
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

1. Run the training script:
   ```bash
   python train.py
   ```

1. Run the inference script:
   ```bash
   python inference.py
   ```

## Notes
- Close the image to move to the next input during inference.