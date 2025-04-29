This repo evaluates how useful Large Language Models can be to automate large-scale mappings of coping options to barriers (context: physical activity). The ultimate goal is to prove that LLMs can explain a sufficient amount of variation in user_rating data. This would justify the usage of LLMs on larger scales where user-based ratings can not be obtained anymore.

User-based (aggregated) relevance scores for 1556 combinations (50 barriers x 64 coping options) were obtained based on the dataset from the following paper by [Braun et al. (2024): "An analysis of physical activity coping plans: mapping barriers and coping strategies based on user ratings"](https://biblio.ugent.be/publication/01JETMFF384JK61N3GAPKG6QGA).

# 1. As predicted: User ratings and LLM ratings of relevance score for coping options, given barriers, are moderately correlated (M(|r|)=0.46):
![Correlation Matrix full dataset](https://github.com/user-attachments/assets/70275761-9a07-4e75-b700-0c9da7fe777a)

# 2. Scatterplots were also created, and fitted to simple linear regression model:
![AlwaysUSER_OverallLLM](https://github.com/user-attachments/assets/faeae849-7053-4569-a2e3-7f384a7b1096)
![NeverUSER_OverallLLM](https://github.com/user-attachments/assets/20d9f7a2-ecd0-4117-86e9-c933ba957cf9)

# 3. Furthermore, a confusion matrix can be constructed with binary relevance estimates. The false negative rate is '16.27%' (196/1205):
![confusion_matrix_gpt4omini](https://github.com/user-attachments/assets/5ff58c82-6201-4a88-b67c-2bbce87eb5d3)

# 4. Both matrix types were also computed seperately for each barrier group and coping option group 
see directory 'images/separate' for (preliminary) results

# 5. Overall accuracies aggregated for different groups and types were also calculated:
  BARRIER GROUPS
  - [INFO] barrier_group 'Bodily and affective feelings' accuracy = 82.50%
  - [INFO] barrier_group 'Capability' accuracy = 70.00%
  - [INFO] barrier_group 'Motivation, beliefs and goal conflict' accuracy = 76.07%
  - [INFO] barrier_group 'Behavioral opportunity' accuracy = 66.00%
  
  COPING OPTION GROUPS
  - [INFO] coping_group 'review behavioral plan' accuracy = 76.47%
  - [INFO] coping_group 'social support, information and awareness' accuracy = 63.64%
  - [INFO] coping_group 'prepare for activity' accuracy = 81.43%
  - [INFO] coping_group 'goal directed BCT' accuracy = 78.26%

# 6. Additional note: This approach likely underestimates the performance of LLMs ; this is likely because of quality issues with the user_rating dataset:
  1. Participants never judged relevance directly; they always received a fixed set of 10 coping options per barrier. This design can introduce bias, whereas the language model assesses relevance directly given a B-C combination.
  2. The user‑rating dataset provides an average of 19.193 data points per relevance score (min=14, max=23), and those responses are already binary --> estimated statistical power of 15%-35% per relevance estimation
  3. Ratings were collected with Qualtrics questionnaires, a subjective method that can be less reliable.
  4. The LLM‑based prediction pipeline is still un‑optimized (model choice, prompt engineering using prompt ablation experiment, currently single‑shot, no additional conditional logic, etc.).
  5. For the estimation of the binary confusion matrix, some misclassifications can be explained by 'suboptimal' user_data category mapping logic (see rows 43, 50, 51, 52, 53, 55, 56, 57, 62, 64, etc.) 
