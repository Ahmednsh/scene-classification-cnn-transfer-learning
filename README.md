# Scene Classification with CNNs — From Scratch vs. True Transfer Learning

Six-class scene classification (buildings · forest · glacier · mountain · sea · street) comparing a custom CNN trained from scratch against DenseNet121 with proper transfer learning — on the full Intel Image Classification dataset (~14k train / 3,000 test images).

**Results: custom CNN from scratch 90.4% · DenseNet121 transfer + fine-tune 92.3%** (full 3,000-image test set, per-class precision/recall/F1 in the notebook).

**Origin & attribution:** based on Lab 6 of the **KAUST Academy Advanced AI program** (with NTDP & MCIT). The original lab structure and dataset belong to the program; the analysis, corrections, and improvements below are my own extension of that coursework.

## What I fixed and improved

| # | Original lab | Improved version | Why it matters |
|---|---|---|---|
| 1 | "Pretrained" DenseNet121 built with `weights=None` — random init, **no transfer learning actually happened** | `weights="imagenet"`, frozen base + custom head, then staged fine-tuning at LR 1e-5 | The core concept of the lab, actually implemented |
| 2 | `Flatten → Dense(4096)` head ≈ **102M parameters** on 1,200 images | `GlobalAveragePooling2D + Dropout(0.3)` | Removes ~100M params → dramatically less overfitting |
| 3 | No data augmentation | Flips / rotations / zooms / shifts on **train only** | Cheapest regularisation for small datasets |
| 4 | Fixed 64 blind epochs | EarlyStopping + ReduceLROnPlateau + ModelCheckpoint | Training becomes a controlled process |
| 5 | Confusion matrix on a **single 60-image batch** | Full test-set evaluation + per-class precision/recall/F1 | One batch can be wildly unrepresentative |
| 6 | Only NumPy seeded; string-default optimizers | `tf.keras.utils.set_random_seed`, explicit LRs | Reproducibility |
| 7 | Lab mini-split: **46 test images with one class absent** — produced inverted results (scratch 43% "beating" transfer 24%) | Diagnosed the data problem, switched to the full Intel dataset via `kagglehub` (~14k/3k) | Results went from statistical noise to 90%+ signal |

Every change is marked `### IMPROVED` in the notebook with an explanation of the reasoning.

## Stack
TensorFlow / Keras · DenseNet121 · scikit-learn (evaluation) · Google Colab (GPU)

## Author
**Ahmed Sharahili** — AI & Full-Stack Engineer
[GitHub](https://github.com/Ahmednsh) · [LinkedIn](https://linkedin.com/in/ahmed-sharahili) · Production work: [Smart Saudi Advisor](https://smart-saudi-advisor.onrender.com) · [Estibsar](https://estibsar.app) · [GO SOUTH KSA](https://go-south-ksa.onrender.com)
