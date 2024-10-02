# auto_labelling_with_vlms
![VLMS](assets/vlms_labelling.png)

Repo to obtain outputs from PaliGemma a Visual Language Model for object detection tasks and using the predictions as labels, visualized through VIA tool by VGG group.

Steps: 
1. Install dependencies ```pip3 install -r requirements.txt```
2. Get token from Hugging Face and set as env variable ``` os.environ["HUGGINGFACE_API_TOKEN"] = "<enter the token here>" ```
3. Put the images for which labels are needed in the images folder
4. Execute ```python3 auto_labelling_paligemma.py```
5. Download the VIA tool : https://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-2.0.12.zip
   <img width="767" alt="Screenshot 2024-09-30 at 10 54 14 PM" src="https://github.com/user-attachments/assets/28dfe030-178d-475b-a0dd-2bca439ea8df">
6. Click on via.html and upload the annotations generated
   <img width="1135" alt="Screenshot 2024-09-30 at 10 50 43 PM" src="https://github.com/user-attachments/assets/2724c542-194d-4732-a95c-901fed0128e3">
7. Make sure to have the images folder inside the via folder as shown in ```step 4```
8. Adjust/Add/Delete the annotations based on the need
   <img width="1725" alt="Screenshot 2024-09-30 at 11 43 25 PM" src="https://github.com/user-attachments/assets/bfbc1d79-0e2e-4484-a272-3b3614754c62">

References: <br>
[1] https://github.com/NSTiwari/PaliGemma <br>
[2] https://huggingface.co/docs/transformers/main/en/model_doc/paligemma
[3] https://www.kaggle.com/datasets/kylegraupe/wind-turbine-image-dataset-for-computer-vision
[4] https://www.robots.ox.ac.uk/~vgg/software/via/


Google Cloud credits are provided for this project #AISprint

## Cite This Work

If you use this project in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{Bhat2024,
  author       = {Rajesh Shreedhar Bhat},
  title        = {Auto Labelling with Vision-Language Models},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/rajesh-bhat/auto_labelling_with_vlms}},
}
