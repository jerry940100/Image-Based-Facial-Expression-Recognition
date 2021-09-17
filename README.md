# Facial-Expression-Recognition


> This note introduces  **ViT_SE**, **Difference of Embeddings**, and **Disentangled Diffrence of Embeddings** models on FER task.

## :memo: Contents

* ### Experimental model introduction

    -  ViT_SE
    -  Difference of Embeddings
    -  Disentangled Difference of Embeddings
* ### Experimental results

## :memo: Experimental model introduction
* ### ViT_SE
    > [Learning Vision Transformer with Squeeze and Excitation for Facial Expression Recognition](https://arxiv.org/abs/2107.03107v3)  
作者提出Vison transformer會逐漸從global attention轉變為local attention，於是作者決定加入SE block來以重新調整local attention features之間的關聯性。
    #### Model structures
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
    #### Model structures
    ![](https://i.imgur.com/Myg15jQ.jpg)

    ##### **Difference_of_Embeddings model Input shape:[Batchsize, 2, C<sub>in</sub>, Height, Width]**
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
    #### Model structures
    ![](https://i.imgur.com/QEH7al2.png)
    ##### **[Pretrained_Emotion_Encoder model](https://github.com/jerry940100/Facial-Expression-Recognition/blob/b1207c23c68fa3fb69e2b7f94ce6241d330b04b3/Disentangled_Difference_of_Embeddings/Pretrained_Emotion_Encoder.py) Input shape:[Batchsize, C<sub>in</sub>, Height, Width]**
    ##### **[Disentangled_Difference_of_Embeddings model](https://github.com/jerry940100/Facial-Expression-Recognition/blob/b1207c23c68fa3fb69e2b7f94ce6241d330b04b3/Disentangled_Difference_of_Embeddings/Disentangled_Difference_of_Embeddings.py) Input shape:[Batchsize, 2, C<sub>in</sub>, Height, Width]**
    
    #### How to implement?
    >This model is trained in 2 steps  
    >1. Train **[Pretrained_Emotion_Encoder](https://github.com/jerry940100/Facial-Expression-Recognition/blob/b1207c23c68fa3fb69e2b7f94ce6241d330b04b3/Disentangled_Difference_of_Embeddings/Pretrained_Emotion_Encoder.py)** by using concatenation of  Identity embedding and Emotion embedding to classify the expressions.*(The [pretrained_Identity_Encoder](https://github.com/cydonia999/VGGFace2-pytorch.git) is based on ResNet50 trained on MS1M and fine-tuned on VGGFace2)*  
    >2. Use Pretrained Emotion Encoder to infer the embeddings of **Neutral image** and **Target expression image**, and then concatenate the difference of embeddings and target expression embedding to infer the class of target expression image.

## :memo: Experimental results

