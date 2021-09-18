# Facial-Expression-Recognition


> This note introduces  **ViT_SE**, **Difference of Embeddings**, and **Disentangled Diffrence of Embeddings** models on FER task.

## :memo: Contents

* ### Experimental Model Introduction

    -  ViT_SE
    -  Difference of Embeddings
    -  Disentangled Difference of Embeddings
* ### Experimental Results
    * Dataset
        * AffectNet small
        * Camera two of CTBC Dataset (collected by MISLAB)
    * Experimental Results of AffectNet small
    * Experimental Results of CTBC Dataset
    * Ablation Study of *Difference of Embeddings* Model

## :memo: Experimental Model Introduction
* ### ViT_SE
    > [Learning Vision Transformer with Squeeze and Excitation for Facial Expression Recognition](https://arxiv.org/abs/2107.03107v3)  
作者提出Vison transformer會逐漸從global attention轉變為local attention，於是作者決定加入SE block來以重新調整local attention features之間的關聯性。
    #### Model Structure
    <img src=https://i.imgur.com/HzOxfgd.png alt="drawing" style="width:300px;vertical-align:middle;"/><br>
    ##### **Input shape:[Batch_size, 3, 224, 224]**
    #### How to implement?
    ##### 1. Install Transformer
    ```
    pip install transformers
    ```
    ##### 2. Read the following resouce
    >Then, you can click the [official introduction pages](https://huggingface.co/transformers/model_doc/vit.html) to learn the API of ViT model, or you can see the introduction and implementation [here](https://hackmd.io/@L-6kLln4ROKxmsbgf1Wt3g/rJ8GpQlJY). Otherwise, you can directly refer the [source code](https://github.com/jerry940100/Facial-Expression-Recognition/blob/5bc742aac221648f8d9c0876dbeba30a39e49932/ViT_SE/ViT_SE.py).
    

* ### Difference of Embeddings
    >This model is going to use difference of embeddings (between Neutral and target expressions) and target expressions embeddings to classify the target expression.
    #### Model Structure
    ![](https://i.imgur.com/Myg15jQ.jpg)

    ##### **Difference_of_Embeddings model Input shape:[Batchsize, <font color=red>2</font>, C<sub>in</sub>, Height, Width]**
        You have to input 2 images(Neutral and target expressions image)
        into the model at the same time.

    #### How to implement?
    >Use Feature Extractor which is based on MobileNetV3 large and pretrained on ImageNet dataset to infer the embeddings of **Neutral image** and **Target expression image**, and then concatenate the difference of embeddings and target expression embedding to infer the class of target expression image.
    >
    >***You can refer the source code [here](https://github.com/jerry940100/Facial-Expression-Recognition/blob/dbd74ea4cfee0867673306b87f6ec8e5d4563a57/DifferenceofEmbeddings/DifferenceofEmbeddings.py).***
    
*  ### Disentangled Difference of Embeddings 
    >This model is going to separate into 2 parts.  
    >1. The first part is going to learn the **Emotion feature extractor** which **aims to ignore the identity of difference people**. 
    >2. The second part is going to use the Emotion feature extractor to **get the emotion embeddings** of **Neutral image** and **Target expression image**. After getting the embeddings, we can calculate the **difference of two embeddings** which can represent the variation between Neutral and Target emotion and **concatenate the difference with the embedding of target expression** to classify the Target expression image.
    #### Model Structure
    ![](https://i.imgur.com/QEH7al2.png)
    ##### **[Pretrained_Emotion_Encoder model](https://github.com/jerry940100/Facial-Expression-Recognition/blob/b1207c23c68fa3fb69e2b7f94ce6241d330b04b3/Disentangled_Difference_of_Embeddings/Pretrained_Emotion_Encoder.py) Input shape:[Batchsize, C<sub>in</sub>, Height, Width]**
    ##### **[Disentangled_Difference_of_Embeddings model](https://github.com/jerry940100/Facial-Expression-Recognition/blob/b1207c23c68fa3fb69e2b7f94ce6241d330b04b3/Disentangled_Difference_of_Embeddings/Disentangled_Difference_of_Embeddings.py) Input shape:[Batchsize, <font color=red>2</font>, C<sub>in</sub>, Height, Width]**
    
    #### How to implement?
    >This model is trained in 2 steps  
    >1. Train **[Pretrained_Emotion_Encoder](https://github.com/jerry940100/Facial-Expression-Recognition/blob/b1207c23c68fa3fb69e2b7f94ce6241d330b04b3/Disentangled_Difference_of_Embeddings/Pretrained_Emotion_Encoder.py)** by using concatenation of  Identity embedding and Emotion embedding to classify the expressions.*(The [pretrained_Identity_Encoder](https://github.com/cydonia999/VGGFace2-pytorch.git) is based on ResNet50 trained on MS1M and fine-tuned on VGGFace2)*  
    >2. Use Pretrained Emotion Encoder to infer the embeddings of **Neutral image** and **Target expression image**, and then concatenate the difference of embeddings and target expression embedding to infer the class of target expression image.

## :memo: Experimental Results
* ### Dataset
    #### 1. AffectNet small
    > [AffectNet](http://mohammadmahoor.com/affectnet/) is an annotated dataset collected in the wild and contains more than 1M facial images. In this experiment, We **only use the sample images from manually annotated part of the AffectNet**, and these images consists of 8 discrete facial expressions.  
    ##### Preprocessing procedures
    1. Random **sample 10% pictures** of eight labels from original manually annotated AffectNet
    2. Crop the face region
    3. Resize to 224×224
    
    ##### Training and Testing data
    

    |  Expressions  | Neutral | Happiness | Sadness | Surprise |
    |:-------------:|:-------:|:---------:|:-------:|:--------:|
    | Training data |  7487   |   13441   |  2545   |   1409   |
    | Testing data  |    500  |    500    |  500    |   500    |

    |  Expressions  | Fear | Disgust | Anger | Contempt |
    |:-------------:|:----:|:-------:|:-----:|:--------:|
    | Training data | 637  |   380   | 2488  |   375    |
    | Testing data  | 500  |  500    |500    |   500    |
    
    #### 2. CTBC
    > CTBC is an annotated dataset consisting of 7 labels and 10 subjects collected by [MISLAB](http://mislab.cs.nthu.edu.tw/). Due to the RAM problems of Colab, We use part of the front face images(camera2) to train our model in some experiments.
    ##### Preprocessing procedures of [Experimental Results of CTBC Dataset](#Expereimental-Results-of-CTBC-Dataset)
    1. Sample 16081 cam2 images
    2. Resize to 224×224
    
    ##### Preprocessing procedures of [Ablation Study of *Difference of Embeddings* Model](#Ablation-Study-of-Difference-of-Embeddings-Model)
    1. Sample all of cam2 data 
    2. Resize to 128×128



*    ### Expereimental Results of AffectNet small
|     Model     |                ViT_SE                |       Different_of_Embeddings        | Disentangled_Difference_of_Embeddings |
|:-------------:|:------------------------------------:|:------------------------------------:|:-------------------------------------:|
| Test Accuracy |                 34.7                 |                 33.6                 |     <font color=red>38.04</font>      |
|    Record     | ![](https://i.imgur.com/1S4egke.png) | ![](https://i.imgur.com/bLubcr7.png) |   ![](https://i.imgur.com/bfwrIYe.png)|

		

*    ### Expereimental Results of CTBC Dataset

*    ### Ablation Study of [Difference of Embeddings](#Difference-of-Embeddings) Model
    
    
 

