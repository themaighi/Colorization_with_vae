## Computer Vision Project


### Project member

- Michele Cavazza (michele.cavazza.1@studenti.unipd.it)


Approaches

### CromoGAN

@article{vitoria2019chromagan,
  title={ChromaGAN: An Adversarial Approach for Picture Colorization},
  author={Vitoria, Patricia and Raad, Lara and Ballester, Coloma},
  journal={arXiv preprint arXiv:1907.09837},
  year={2019}
}

## Dataset

Dataset: https://www.kaggle.com/moltean/fruits

## Proposal

Topic: Grayscale Image Colorization

### What is the problem that you will be investigating? 
Our task is to colorize grayscale images. We will use two different approaches to reach our goal. Their performances will be compared between each other and the ground truth. The first approach is based on CNN and the second is based on GAN.

### What method or algorithm are you proposing? If there are existing implementations, will you use them and how? 

![alt text](https://raw.githubusercontent.com/pvitoria/ChromaGAN/master/Figures/ColorizationModel.png)
 
The picture above shows the architecture for the second approach. The GAN architecture contains two sub architecture:
Generator architecture, this architecture is composed of two subnetworks. The first subnetwork (composed by the yellow, purple red and blue part)  will output the crominance of the image, and the second one (composed by the yellow, red and grey part) will output the class distribution vector. As it can be seen the two subnetworks share some modules.
Discriminator architecture, this part of the network will be based on PatchGan that focuses on detecting if each image patch is real or fake.
 


### How do you plan to modify such implementations? 

We are going to freeze the weights of the already pre-trained Chromagan and remove the Gan component of it. Instead, we will add a VAE component
