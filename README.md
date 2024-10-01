# labelling_with_paligemma
Repo to obtain output from PaliGemma for object detection tasks and using the predictions as labels visualized through VIA tool by VGG group. 


Steps: 
1. Get token from Hugging Face and set as env variable. 
2. Put the images for which labels are needed in the images folder
3. Execute ```python3 paligemma_to_labels.py --image-dir="images"```
4. Download the VIA tool
5. Upload the annotations.
6. Adjust the annotations if needed
