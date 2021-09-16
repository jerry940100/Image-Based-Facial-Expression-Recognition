# Facial-Expression-Recognition


> This note introduces  **ViT_SE**, **Difference of Embeddings**, and **Disentangled Diffrence of Embeddings** models on FER task.

## :memo: Contents

* ### Experimental model introduction

    -  ViT_SE
    -  Difference of Embeddings
    -  Disentangled Difference of Embeddings
* ### Experimental result

## :memo: Experimental model introduction
* ### ViT_SE
    > [Learning Vision Transformer with Squeeze and Excitation for Facial Expression Recognition](https://arxiv.org/abs/2107.03107v3)  
作者提出Vison transformer可以學到local feature，於是作者決定加入SE block學習global attention relation，來重新調整不同local feature之間的關係。
    #### Model structures
    <img src=https://i.imgur.com/HzOxfgd.png alt="drawing" style="width:200px;vertical-align:middle;"/><br>
    #### Install Transformer
    ```
    pip install transformers
    ```
    Then, you can click the [official introduction pages](https://huggingface.co/transformers/model_doc/vit.html) to learn the API of ViT model, or you can see the introduction and implementation [here](https://hackmd.io/@L-6kLln4ROKxmsbgf1Wt3g/rJ8GpQlJY). Otherwise, you can directly refer the [source code](https://github.com/jerry940100/Facial-Expression-Recognition/blob/5bc742aac221648f8d9c0876dbeba30a39e49932/ViT_SE/ViT_SE.py).
    

* ### Difference of Embeddings
    >This model is going to use difference of embeddings (between Neutral and target expressions) and target expressions embeddings to learn the expression classification.
    #### Model structures
    ![](https://i.imgur.com/B3yKcqv.png)
    ##### **Input shape:[Batch_size,2,3,224,224]**
        You have to input 2 images(Neutal and target expressions image)
        into the model at the same time.
    

    


Let's try it out!
Apply different styling to this paragraph:
**HackMD gets everyone on the same page with Markdown.** ==Real-time collaborate on any documentation in markdown.== Capture fleeting ideas and formalize tribal knowledge.

- [x] **Bold**
- [ ] *Italic*
- [ ] Super^script^
- [ ] Sub~script~
- [ ] ~~Crossed~~
- [x] ==Highlight==

:::info
:bulb: **Hint:** You can also apply styling from the toolbar at the top :arrow_upper_left: of the editing area.

![](https://i.imgur.com/Cnle9f9.png)
:::

> Drag-n-drop image from your file system to the editor to paste it!

### Step 3: Invite your team to collaborate!

Click on the <i class="fa fa-share-alt"></i> **Sharing** menu :arrow_upper_right: and invite your team to collaborate on this note!

![permalink setting demo](https://i.imgur.com/PjUhQBB.gif)

- [ ] Register and sign-in to HackMD (to use advanced features :tada: ) 
- [ ] Set Permalink for this note
- [ ] Copy and share the link with your team

:::info
:pushpin: Want to learn more? ➜ [HackMD Tutorials](https://hackmd.io/c/tutorials) 
:::

---

## BONUS: More cool ways to HackMD!

- Table

| Features          | Tutorials               |
| ----------------- |:----------------------- |
| GitHub Sync       | [:link:][GitHub-Sync]   |
| Browser Extension | [:link:][HackMD-it]     |
| Book Mode         | [:link:][Book-mode]     |
| Slide Mode        | [:link:][Slide-mode]    | 
| Share & Publish   | [:link:][Share-Publish] |

[GitHub-Sync]: https://hackmd.io/c/tutorials/%2Fs%2Flink-with-github
[HackMD-it]: https://hackmd.io/c/tutorials/%2Fs%2Fhackmd-it
[Book-mode]: https://hackmd.io/c/tutorials/%2Fs%2Fhow-to-create-book
[Slide-mode]: https://hackmd.io/c/tutorials/%2Fs%2Fhow-to-create-slide-deck
[Share-Publish]: https://hackmd.io/c/tutorials/%2Fs%2Fhow-to-publish-note

- LaTeX for formulas

$$
x = {-b \pm \sqrt{b^2-4ac} \over 2a}
$$

- Code block with color and line numbers：
```javascript=16
var s = "JavaScript syntax highlighting";
alert(s);
```

- UML diagrams
```sequence
Alice->Bob: Hello Bob, how are you?
Note right of Bob: Bob thinks
Bob-->Alice: I am good thanks!
Note left of Alice: Alice responds
Alice->Bob: Where have you been?
```
- Auto-generated Table of Content
[ToC]

> Leave in-line comments! [color=#3b75c6]

- Embed YouTube Videos

{%youtube PJuNmlE74BQ %}

> Put your cursor right behind an empty bracket {} :arrow_left: and see all your choices.

- And MORE ➜ [HackMD Tutorials](https://hackmd.io/c/tutorials)
