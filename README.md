# auto_labelling_with_vlms
![VLMS](assets/vlms_labelling.png)

Repo to obtain outputs from PaliGemma a Visual Language Model for object detection tasks and using the predictions as labels, visualized through VIA tool by VGG group. 
![paligemma_arch](https://github.com/user-attachments/assets/9d48def5-5f2b-4d6a-998b-956697e9f011)

Steps: 
1. Get token from Hugging Face and set as env variable ``` os.environ["HUGGINGFACE_API_TOKEN"] = "<enter the token here>" ```
2. Put the images for which labels are needed in the images folder
3. Execute ```python3 auto_labelling_paligemma.py```
4. Download the VIA tool : https://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-2.0.12.zip
   <img width="767" alt="Screenshot 2024-09-30 at 10 54 14 PM" src="https://github.com/user-attachments/assets/28dfe030-178d-475b-a0dd-2bca439ea8df">
6. Click on via.html and upload the annotations generated
   <img width="1135" alt="Screenshot 2024-09-30 at 10 50 43 PM" src="https://github.com/user-attachments/assets/2724c542-194d-4732-a95c-901fed0128e3">
7. Make sure to have the images folder inside the via folder as shown in ```step 4```
8. Adjust/Add/Delete the annotations on a need
   <img width="1725" alt="Screenshot 2024-09-30 at 11 43 25 PM" src="https://github.com/user-attachments/assets/bfbc1d79-0e2e-4484-a272-3b3614754c62">

References: <br>
[1] https://github.com/NSTiwari/PaliGemma <br>
[2] https://huggingface.co/docs/transformers/main/en/model_doc/paligemma
