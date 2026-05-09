# Sleep-Stage-Classification-XAI
This project focuses on replicating [a paper about predicting sleep stages](https://ieeexplore.ieee.org/abstract/document/11429301) published in the IEEE International Conference on Electrical, Computer & Telecommunication Engineering in 2026.
In general terms, it consists on analysing the public dataset [DREAMT](https://physionet.org/content/dreamt/1.0.0/), which contains loads of information extracted by wearable devices about sleep based features, such as heartbeat rate (BVP), temperature, acceleration on each axe and even sleep related diseases, among others. The process consists on:
1. Perform an initial data cleaning and standarize wave patterns to ensure a proper analysis on following steps.
2. Extract features, mostly based on the BVP column, that will be used in the prediction stage.
3. Classify the current stage, given the different phases (sleep VS. wake, wake VS. NREM VS. REM, wake vs. light sleep vs. deep sleep vs. REM)
   
After that, some Explainable Artificial Intelligence are applied, in order to determine which components play the most crucial role when it comes to predicting, providing a huge value on tracing the model's decision patterns. Here is where the added value to this work comes, as a deeper analysis on this stage will be made to provide valuable insights and improve the general knowledge about how this model works.
