# auto_labelling_with_vlms
Repo to obtain output from PaliGemma for object detection tasks and using the predictions as labels visualized through VIA tool by VGG group. 


Steps: 
1. Get token from Hugging Face and set as env variable. 
2. Put the images for which labels are needed in the images folder
3. Execute ```python3 auto_labelling_paligemma.py --image-dir="images"```
4. Download the VIA tool : https://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-2.0.12.zip
5. Upload the annotations
   <img width="1135" alt="Screenshot 2024-09-30 at 10 50 43 PM" src="https://github.com/user-attachments/assets/2724c542-194d-4732-a95c-901fed0128e3">
6. Make sure to have the images folder inside the via folder. Example
   <img width="767" alt="Screenshot 2024-09-30 at 10 54 14 PM" src="https://github.com/user-attachments/assets/28dfe030-178d-475b-a0dd-2bca439ea8df">
7. Adjust/Add/Delete the annotations on a need 
