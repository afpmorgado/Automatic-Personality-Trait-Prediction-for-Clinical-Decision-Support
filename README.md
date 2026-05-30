Repository for the source code of the Master's Thesis "Automatic Personality Trait Prediction for Clinical Decision Support", Instituto Superior Técnico, Universidade Lisboa, 2026.

Author: André Morgado

Results of the thesis' experiments with the respective weights of the trained models: https://drive.google.com/drive/folders/1ieqp1rnnzUNqtLTkNxcyW26UTBWecvA9?usp=sharing

Abstract: "Personality trait assessment provides clinicians with insights into patient behavior, communication patterns and potential treatment responses. Traditionally, it relies on self-reported questionnaires which are time consuming and subject to biases, such as variance in self-perception. Advancements in natural language processing and deep learning, fueled by the availability of data, have enabled automated text-based assessment, addressing these limitations.

This thesis proposes a DistilBERT-based model for personality trait prediction evaluated on the Essays (Big-Five) and MBTI Kaggle datasets. A two-phase experimental study was conducted: an ablation study of architectural components including psycholinguistic features, long-context embeddings and Focal Loss, followed by an evaluation of chunk pooling strategies to combine textual segments while overcoming DistilBERT's input token limits. Two preprocessing pipelines were also compared.

Results show that effectiveness depends on dataset characteristics. Focal Loss improves Macro-F1 on the imbalanced MBTI dataset but degrades performance on Essays. Additional features generally did not improve results. Attention Pooling benefits fragmented texts but is less effective for long essays where chunking reduces contextual continuity. A lighter preprocessing pipeline performed better in both datasets.

Final models achieved 56.1% accuracy on Essays and 72.1% Macro-F1 on MBTI, below state-of-the-art by 11.7% and 7.3%, although boasting a lower computational cost. Beyond standard metrics, reliability was analyzed via confusion matrices, calibration, Brier score, robustness to paraphrasing, and sensitivity to text length.

Overall, results demonstrate strong performance on the Kaggle MBTI dataset, highlighting a favorable trade-off between computational efficiency and predictive performance, as well as the importance of reliability evaluation preceding clinical deployment of the model."
