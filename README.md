# :construction: social_lstm_tf :construction:

Social LSTM implementation with TensorFlow.

Important Notes:

* **Unofficial and experimental implementation.**
* (2018/12/27) I have been updating my codes.
* I have been rewriting all of my codes without any other codes. Therefore I have changed the repo license to MIT License.

## Requirements

Works only Python 3. My environment is built on [Anaconda](https://www.anaconda.com/download/). Please install it and the following packages.

* [TensorFlow 1.12.0](https://github.com/tensorflow/tensorflow)
* [patool](https://github.com/wummel/patool): to extract .rar file

## :construction: Usage :construction:

The following steps are not correctly updated.

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

**The batch size must be 1.** It is difficult for me to handle trajectory samples having different shape in a batch.

## References

* Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, Silvio Savarese. **Social LSTM: Human Trajectory Prediction in Crowded Spaces**. The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 961-971
* My implementation based on: [Social LSTM using TensorFlow](https://github.com/vvanirudh/social-lstm-tf)