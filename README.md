# social_lstm_keras_tf

**2019/01/22 - now Please See `renew` branch.**

Social LSTM implementation with Keras (and TensorFlow as backend)  
NOTE: experimental implementation

## Requirements (show primary packages only)

* Python 3.6.0+
* Keras 2.1.5+
* TensorFlow 1.6.0+

## Usage

### 1. Preparation

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

## References

* Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, Silvio Savarese. **Social LSTM: Human Trajectory Prediction in Crowded Spaces**. The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 961-971
* My implementation based on: [Social LSTM using TensorFlow](https://github.com/vvanirudh/social-lstm-tf)
