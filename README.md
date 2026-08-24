# ODE-Transformer for Sign Language Recognition

This repository contains the official PyTorch implementation of the experiments conducted for my thesis named **An Iterative Transformer Framework for Isolated Turkish Sign Language Recognition**. 

This work introduces an **ODE (Ordinary Differential Equation) Transformer** implementation built on top of the framework originally developed for the [ChaLearn 2021 LAP Challenge](https://github.com/m-decoster/ChaLearn-2021-LAP).

## 📖 About the Project & Dataset
This codebase contains the experimental setup and results for a Master's Thesis focusing on **Turkish Sign Language Recognition**. The experiments were exclusively conducted on the **AUTSL (Ankara University Turkish Sign Language)** dataset, which is a large-scale, high-quality dataset containing 226 distinct signs performed by 43 different signers.

### The Role of the ODE-Transformer
Sign language is highly continuous and dynamic; meaning standard discrete attention mechanisms (like those in a vanilla Video Transformer Network) may struggle to capture the continuous flow of complex hand and body motions. To address this, we integrated an **ODE-Transformer**:
- It models the hidden states of the neural network as a continuous-time continuous-depth physical system.
- By using differential equation solvers (like Runge-Kutta), the network gains a smoother and more robust understanding of temporal dependencies in video frames.
- As demonstrated in the results below, replacing standard discrete layers with ODE-based attention significantly boosts recognition accuracy on the AUTSL dataset.

## 📢 Acknowledgements & References
This codebase is a modified extension of the original repository. 

* **Original Repository:** [m-decoster/ChaLearn-2021-LAP](https://github.com/m-decoster/ChaLearn-2021-LAP)

If you use this code for your research, please consider citing the thesis and the original work:

```bibtex
@mastersthesis{isik2026iterative,
  author  = {Işık, Ö. F.},
  title   = {An Iterative Transformer Framework for Isolated {Turkish} Sign Language Recognition},
  school  = {Hacettepe Üniversitesi, Fen Bilimleri Enstitüsü},
  year    = {2026},
  type    = {Yüksek Lisans Tezi},
  address = {Ankara, Türkiye}
}

@InProceedings{De_Coster_2021_CVPR,
    author    = {De Coster, Mathieu and Van Herreweghe, Mieke and Dambre, Joni},
    title     = {Isolated Sign Recognition From RGB Video Using Pose Flow and Self-Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {3441-3450}
}
```
## ✨ Novel Contributions
In this repository, we extended the original Video Transformer Network (VTN) by introducing:
* **ODE Transformer integration** for continuous-time temporal modeling.
* Modified `src/models/` architecture supporting `rk_type` and `encoder_history_type` arguments.
* Customized training loop and dynamic hyperparameter configuration via `src/config.sh` and `src/config_paths.py` for optimized memory management and easy local path handling.

## 📊 Experimental Results
The integration of the ODE-Transformer yields improvements over the standard VTN baseline. Below is a comparative table of the best results achieved:

| Model Configuration | `encoder_calculate_num` | `rk_type` | `encoder_history_type` | Accuracy  |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Standard VTN)** | `baseline` | `none` | `none` | **93.05%**  |
| ODE-Transformer | `rk2` | `none` | `none` | 93.26%  |
| **ODE-Transformer (Best)** | `rk2` | `none` | `dense` | **93.99%** |

*Results obtained from the last checkpoint on the test set.*

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r src/requirements.txt
```

### Dataset Structure
Ensure your dataset is placed under the `data/` directory. The codebase expects the following structure by default:
- `data/mp4/`
- `data/kp/`
- `data/kpflow2/`

*Note: All paths are dynamically handled via `src/config_paths.py`. If you change the data location, simply update the paths in `config_paths.py` without modifying the core logic.*

### Training the Model
We provide an easy-to-use bash script to configure hyperparameters (such as batch size, learning rate, and ODE configurations) and run the training pipeline:

1. Edit the hyperparameters in `src/config.sh`
2. Run the experiment:
```bash
cd src
python run_experiment.py
```

## 📜 License
This project inherits the license of the original ChaLearn-2021-LAP repository. Please see the `LICENCE` file for details.
