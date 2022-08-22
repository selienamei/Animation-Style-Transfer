# Animation style transfer using Deep-Learning models, cycleGAN
The goal of this project is to see if deep learning models can learn the animation styles of different studios or mediums. This project use animations from Pixar and 
Studio Ghibli. Pixar films use 3-Dimensional (3D) animation and Studio Ghibli uses 2-Dimensional (2D) animation.
<br />
<br />
This is a group project, and I am only working on the cycleGAN part. For more information, there is a final report 
[here](https://github.com/selienamei/Deep-Learning/blob/main/Animation%20Transfer%20Report.pdf).
<br />
The data comes from animation screen captures (i.e.
screenshots) of various animated movies. The data
can be found and downloaded at
https://animationscreencaps.com/
<br />
<br />
## CycleGAN
Animations used in this model: 
<br />
Luca (Pixar 3D) and Spirited Away (Ghibli 2D)
<br />
Toy Story 3 (Pixar 3D) and Ponyo (Ghibli 2D) 
<br />
<br />
For CycleGAN, it needs to have 2 generators and 2
discriminators. For the generator, this model uses U-Net-based
architecture. CycleGAN is very similar to CoGan,
also called PixtoPix. The difference is CycleGAN
uses instance normalization, and CoGAN uses batch
normalization. CycleGAN also included two more
loss functions besides generator loss and
discriminator loss. 
<br />
<br />
This project uses the Kaggle environment to train the
CycleGAN model with a GPU. When training this model, it took 7-9 hours. Since the Kaggle environment maximum hours for each session is 9 hours, this project can
only handle 10 epochs. The images below are not perfect, but it is working. The result will be better if the model trains with more epochs. 
<br />
### Result
Toy Story to 2D:
<br />
![alt text](https://github.com/selienamei/Deep-Learning/blob/main/style%20transfer%20images/toy_to_2d_2.png)
<br />
<br />
Ponyo to 3D:
<br />
![alt text](https://github.com/selienamei/Deep-Learning/blob/main/style%20transfer%20images/ponyo_to_3d.png)
<br />
For more images, go to style transfer images folder 

