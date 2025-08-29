
# Waste Classifier

A project to classify waste images as recyclable or organic using transfer learning (VGG16).

## Structure

- `src/train.py`: Training script
- `models/waste_classifier.py`: Model definition
- `tests/test_train.py`: Basic tests
- `data/`: Place your dataset here

## Usage

1. Install requirements:
    ```
    pip install -r requirements.txt
    ```
2. Prepare your data in `data/o-vs-r-split/train` and `data/o-vs-r-split/test`.
3. Train the model:
    ```
    python src/train.py
    ```
4. Run tests:
    ```
    python -m unittest discover tests
    ```

