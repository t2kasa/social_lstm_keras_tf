# social_lstm_keras_tf

Social LSTM implementation with TensorFlow.

Important Notes:

* **Unofficial and experimental implementation.**
* (2018/12/24) I have been updating my codes to correctly train and evaluate Social LSTM.

## Requirements

Works only Python 3. My environment is built on [Anaconda](https://www.anaconda.com/download/). you Install it and the following packages.

* TensorFlow 1.x
* [patool](https://github.com/wummel/patool)

## Usage

### 1. Preparation

Download datasets by:
```bash
cd examples
python prepare_dataset.py
```


Set `dataset` attribute of the config files in `configs/`.

### 2. Training

Run `train_social_model.py`.
```
python train_social_model.py --config path/to/config.json [--out_root OUT_ROOT]
```

### 3. Testing

Run `evaluate_social_model.py`
```
python train_social_model.py --trained_model_config path/to/config.json --trained_model_file path/to/trained_model.h5
```

## Restrictions

* work only on batch size = 1
* require much RAM (use almost all 16GB in my environment)

## TODO

* [ ] Clean up experimental implementation
* [ ] Add trained sample for demo
* [ ] Occupancy map implementation


## References

* Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, Silvio Savarese. **Social LSTM: Human Trajectory Prediction in Crowded Spaces**. The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 961-971
* My implementation based on: [Social LSTM using TensorFlow](https://github.com/vvanirudh/social-lstm-tf)