# Neural-Style-Transfer
This is an implementation of Neural Style Transfer using pretrained VGG19 model (can be found at http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)

The algorithm used for style transfer is based on Andrew Ng's Deep Learning Specialization course on Coursera.

For the content image:

<img src="images/mumbai.jpg" width="400px" height="300px" />

And style image:

<img src="images/paint.jpg" width="400px" height="300px" /><img src="images/guernica.jpg" width="400px" height="300px" />

Generated image:

<img src="output/mumbai+paint.jpg" width="400px" height="300px" /><img src="output/mumbai+guernica.jpg" width="400px" height="300px" />

The generated image differs according to the control parameters like:

For example, generated image for <acau and StarryNight after 200 iterations vs 2000 iterations

<img src="output/Macau+StarryNight200.jpg" width="400px" height="300px" /><img src="output/Macau+StarryNight2000.png" width="400px" height="300px" />
