## Backpropagation
##### An overview
We know that neural networks can learn their weights and biases using the gradient descent algorithm. In this post I will try to discuss how the gradient of the cost function is computed using a fast algorithm known as backpropagation. Backpropagation works much faster than earlier approaches to learning and hence helps the neural network to solve a given problem.It gives us detailed insights into how changing the weights and biases changes the overall behavior of the network.<br>

As per Wiki:<br>
*Backpropagation, an abbreviation for “backward propagation of errors”, is a common method of training artificial neural networks used in conjunction with an optimization method such as gradient descent. The method calculates the gradient of a loss function with respect to all the weights in the network. The gradient is fed to the optimization method which in turn uses it to update the weights, in an attempt to minimize the loss function.*

The simple difference between Forward and backward propagation can be clearly seen in below example.
![](/images/p1.jpg)	
<br>The forward pass above calculates output using the input variables ‘Hours Studied’ and ‘Mid term marks’. The activation function V can be any function. 
![](/images/p2.jpg)	
 
<br>The above figure shows the backward pass. The gradients are computed w.r.t the predicted incorrect output and the weights are updated so as to minimize the error.<br>
In this post I will discuss the backpropagation algorithm in Different neural networks which are :
1.	MLP
2.	CNN
3.	RNN
4.	LSTM
For all these networks, this post will be focused on Backpropagation only. I will not dive deep into forward propagation, chain rule etc, but will use it whenever it is required.
### 1.	Multi Layer Perceptrons
The basic unit of MLP is a perceptron. Perceptron is a type of artificial neuron It takes several binary inputs, x1,x2,…x1,x2,…, and produces a single binary output. It can have more or fewer inputs. In the image below the perceptron has  three inputs, x1,x2, x3.
<br>![](/images/p3.jpg)
<br>![](/images/p4.jpg)
<p>The forwardpass on left calculates ‘z’ as a function of ‘f(x,y)’ using the input variables ‘x’ and ‘y’. The function f can be any function. The right side of the figure shows the backward pass. Receiving ’dL/dz’, the gradients of the loss function w.r.t ‘z’ , the gradients of ‘x’ and ‘y’ on the loss function can be calculated by applying the chain rule.</p>

<p>As the name suggest, a multilayer perceptron (MLP) is a class of feedforward artificial neural network. A MLP consists of at least three layers of nodes: 
i.	one (passthrough) input layer, 
ii.	one or more layers of LTUs, called hidden layers, 
iii.	one final layer of LTUs called the output layer.
Every layer except the output layer includes a bias neuron and is fully connected to the next layer. Also, Except for the input nodes, each node is a neuron that uses a nonlinear activation function.</p>
<br>![](/images/p5.jpg)
<p>
Now, for each training instance the backpropagation algorithm first makes a prediction *(forward pass)*, measures the error, then goes through each layer in reverse to measure the error contribution from each connection *(reverse pass)*, and finally slightly tweaks the connection weights to reduce the error *(Gradient Descent step).*</p>

<p>In the beginning, we initialize weights with some random values(or using suitable weight initialization method), it is not necessary that whatever weight values we have selected will be correct, or it fits our model the best. Hence, our model output is way different than our actual output i.e. the error value is huge. In order to reduce this error, the parameters needs to be modified and for this purpose  Backpropagation is used to train our model.</p>

*	Calculate the error – How far is your model output from the actual output.
*	Minimum Error – Check whether the error is minimized or not.
*	Update the parameters – If the error is huge then, update the parameters (weights and biases). After that again check the error. Repeat the process until the error becomes minimum.
*	Model is ready to make a prediction – Once the error becomes minimum, you can feed some inputs to your model and it will produce the output.

<p>Hence the basic idea behind Backpropagation algorithm is that it looks for the minimum value of the error function in weight space using gradient descent. The weights that minimize the error function is then considered to be a solution to the learning problem.</p>
To summarize: 
1.	Input a set of training examples
2.	For each training example x: Set the corresponding input activation ax,1, and perform the following steps:
*	Feedforward: For each l=2, 3 ,…, L compute   zx,l=wlax,l−1+bl and ax,l=σ(zx,l) 
*	Output error δx,L: Compute the vector δx,L=∇aCx⊙σ′(zx,L)
*	Backpropagate the error: For each l=L−1, L−2, …, 2 compute δx,l=((wl+1)Tδx,l+1)⊙σ′(zx,l).
3.	Gradient descent: For each l=L, L−1,…,2l=L,L−1,…,2 update the weights according to the rule wl→wl−η/m∑xδx,l(ax,l−1)T, and the biases according to the rule bl→bl−η/m∑xδx,l.
<p>Here, b is bias term, l is the layer, C is the cost function<p>
This can be visualized from the below figure.
<br>![](/images/p6.jpg)
<p>Mathematically, I will be using a notation which will let us refer to weights in the network in a simple way. We'll use wjkl to denote the weight for the connection from the kth neuron in the (l−1)th layer to the jth neuron in the lth layer. For example, the diagram below shows the weight on a connection from the fourth neuron in the second layer to the second neuron in the third layer of a network</p>
<br>![](/images/p7.jpg)
 
Similarly, I will use blj for the bias of the jth neuron in the lth layer. And use alj for the activation of the jth neuron in the lth layer.<br>
The activation a in l layer can be related to activation in  (l-1) layer using the equation 
<br>![](/images/p8.jpg)
 
<br>Here, the sum is computed over all neurons k in the (l-1)th layer.<br>
The above equation can be represented in the matrix form as below.
<br>![](/images/p9.jpg) 
 
Hence, we just apply the weight matrix to the activations, then add the bias vector, and finally apply the σ function. This representation is easy to understand than the previous neuron to neuron equation.<br>
Let's imagine that we've made a small change Δwljk to some weight in the network, wljk. Now, this change in weight will cause a change in the output activation from the corresponding neuron Δalj which can be represented as 
<br>![](/images/p10.jpg)
 
<p>This small change will change all the  activations in the next layer:</p>
<br>![](/images/p11.jpg)
 
<p>Combining above 2 equation gives</p> 
<br>![](/images/p12.jpg)
<p>These changes will in turn cause changes in the next layer, and so on all the way through to cause a change in the final layer, and then in the cost function. If the path goes through activations alj, al+1q , … , a L−1n ,aLm then the resulting expression is</p>
<br>![](/images/p13.jpg)
 
<br>That is,
<br>![](/images/p14.jpg)
<p>This represents the change in C due to changes in the activations along this particular path through the network. For above equation we have considered only a single path. However, there are many paths by which change in wjkl can affect the cost function. To compute the total change in C it is plausible that we should sum over all the possible paths between the weight and the final cost, i.e.,</p>
<br>![](/images/p15.jpg)
<p>From above equation, it is clear that in backpropagation, we have a lot of derivative terms that are repeated multiple times. Computing these terms everytime is very costly. Hence it is a good idea to compute these terms once, store them and then use it whenever it is required. Although memory usage is increased but the computation speed increases as we do not have to compute the same derivative again and again. This process is also called as memorization. Hence Backpropagation can be defined as combination of chain rule Chain rule and Memoization </p>
2.	CNN : 
Introduction
Convolutional neural networks (CNNs) are a biologically-inspired variation of the multilayer perceptrons (MLPs). Neurons in CNNs share weights unlike in MLPs where each neuron has a separate weight vector. This sharing of weights ends up reducing the overall number of trainable weights hence introducing sparsity.
Below image shows the transformation of MLP to CNN
 
Utilizing the weights sharing strategy, neurons are able to perform convolutions on the data with the convolution filter. This is then followed by a pooling operation which as a form of non-linear down-sampling, progressively reduces the spatial size of the representation thus reducing the amount of computation and parameters in the network.
In CNN, for each training instance the backpropagation algorithm first makes a prediction which is called as forward pass, it measures the error, and then goes through each layer in reverse to measure the error contribution from each connection, this step is also known as reverse pass, and finally slightly tweaks the connection weights to reduce the error (Gradient Descent step).
The image below shows the forward pass in CNN.
 
Weights are just filters which are also called as kernels, convolution matrices, or masks. The matrix dot products are replaced by convolution operations both in feed forward and backpropagation.
The convolution equation of the input at layer l is given by:
 
Where,
i.	oli,j is the output vector at layer l given by oli,j=f(xli,j)
ii.	f(⋅) is the activation function.
iii.	xli,j is the convolved input vector at layer l plus the bias represented as 
 
iv.	wlm,n is the weight matrix connecting neurons of layer l with neurons of layer l−1.

After the forward pass, for a total of P predictions, the predicted network outputs yp and their corresponding targeted values tp the mean squared error is given by:
 
For the purpose of simplicity, we shall use the case where the input image is grayscale i.e single channel C=1
The output from the convolution procedure is as follows:
 
Please note that the above equation is for C=1. In case when we have 3 channels, the output convolution equation can be represented as below:
 
Learning in this network will be achieved by adjusting the weights such that yp is as close as possible or equals to corresponding tp. In the classical backpropagation algorithm, the weights are changed according to the gradient descent direction of an error surface E. 
 
Backpropagation in CNNs
Backpropagation
We need to perform 2 updates for Backpropagation. For weights and for Deltas. 
We need to compute ∂E/∂wlm′,n′   which can be described as how the change in a single pixel wm′,n′ in the weight kernel affects the loss function E.
In the below image, it is clear that the yellow pixel in kernel makes a contribution in all the products during forward propagation. This means that pixel wm′,n′ will eventually affect all the elements in the output feature map.

 
Hence, Convolution between the input feature map of dimension H×W and the weight kernel of dimension k1×k2 produces an output feature map of size (H−k1+1) by (W−k2+1). The gradient component for the individual weights can be obtained by applying the chain rule in the following way:
 ……..(1)
Putting the value of xli,j (which is the convolved input vector at layer l plus the bias) in above equation gives
 
Further expanding the summations in above equation and taking the partial derivatives for all the components results in zero values for all except the components where m=m′ and n=n′ in wlm,nol−1i+m,j+n
 …(2)
Substituting  eq 2 in eq 1 gives
 
The summations in Eq 3 represents a collection of all the gradients δli,j coming from all the outputs in layer l.
To obtain the gradients w.r.t filter maps, we have a cross-correlation which is transformed into a convolution by flipping the delta matrix. the flipped delta matrix is shown below:
 
The diagram below shows gradients (δ11,δ12,δ21,δ22) generated during backpropagation:
 
Now we will obtain the new set of weights with convolution operation as is shown below:
 
During the reconstruction process, the deltas (δ11,δ12,δ21,δ22) are used. These deltas are provided by an equation of the form:
 
 
From the diagram above, we can see that region in the output which is affected by pixel xi′,j′ from the input is the region in the output bounded by the dashed lines where the top left corner pixel is given by (i′−k1+1,j′−k2+1) and the bottom right corner pixel is given by (i′,j′).
Using chain rule and introducing sums give us the following equation:
 
replacing the value of xl+1i′−m,j′−n  and expanding this part of the equation gives us:
 
taking the partial derivatives for all the components results in zero values for all except the components where m′=m and n′=n, so that f(xli′−m+m′,j′−n+n′) becomes f(xli′,j′) and wl+1m′,n′ becomes wl+1m,n
 
For backpropagation, we make use of the flipped kernel and as a result we will now have a convolution that is expressed as a cross-correlation with a flipped kernel:
 
The above equation is the backpropagation equation in case on CNN. 
3.	Recurrent Neural Networks
The idea behind RNNs is to make use of sequential information. In a traditional neural network, we assume that all inputs and outputs are independent of each other. But for certain tasks such as to predict the next word in a sentence we need to know which words came before it. RNN perform the same task for every element of a sequence, with the output being depended on the previous computations. We can think of RNN’s of having a ‘Memory’, which captures information about what has been calculated so far.
 
the unfolding in time of the computation involved in its forward computation
here,
•	xt is the input at time step t.
•	st is the hidden state at time step t. st=f(Uxt + Wst-1)  where, the function f usually is a nonlinearity such as tanh or ReLU
•	ot is the output at step t.  ot=softmax(Vst)
Backpropagation Through Time
Since RNN is a type of Neural network, learning process or calculation of gradients is also achieved by Backpropagation. In case of RNNs, Backpropagation is called as Backpropagation Through Time (BPTT) and is different from traditional backpropagation. Now as shown above, the basic equation of RNNs is 
st=f(Uxt + Wst-1)
ot=softmax(Vst)
Taking our loss, or error, to be the cross entropy loss.
 
In above equation, yt  is the correct word at time step t, and   is our prediction. Taking the full sequence as one training example, hence the total error is the sum of the errors at each time step (word).
 
We will calculate the gradients of errors w.r.t U,V and W and then learn the best parameters using Stochastic Gradient Descent.
Taking the sum of gradients at each time step for one training example
 
Using chain rule of differentiation and applying the backpropagation algorithm and using E3 as an example.
Gradient w.r.t V:
 
Here, z3  is equal to Vs3
From the above equation, it is clear that     only depends on the values at the current time step  . If we have these, calculating the gradient for V is a simple matrix multiplication.
Gradient w.r.t W:
Applying chain rule we get :
 
However, for E3, s3=tanh(Ux3 + Ws2), ie it depends on s2, which depend on s1 and so on. Hence for taking the derivative W we cannot take treat s2 as a constant. Therefore, we will apply chain rule: 
 
Since, W is used in every step up to the output we care about(till 3), we need to backpropagate gradients from t=3 through the network all the way to t=0:
 
this is exactly the same as the standard backpropagation algorithm that we use in deep Feedforward Neural Networks. The key difference is that we sum up the gradients for W at each time step.
RNNs are hard to train: Sequences (sentences) can be quite long, perhaps 20 words or more, and thus we need to back-propagate through many layers.
4.	LSTM
The most popular model for RNN right now is the LSTM (Long Short-Term Memory) network. LSTM’s are super powerful as Training in LSTM converges faster and it can detect long-term dependencies in the data.
 
A LSTM Cell
Introduction : In LSTM, the cell state is split in two vectors: h(t) and c(t) and h(t) can be thought of as the short-term state and c(t) as the long-term state. 
Now, as the long-term state c(t–1) traverses the network from left to right, it first goes through a forget gate, dropping some memories, and then it adds some new memories via the addition operation which adds the memories that were selected by an input gate. The result c(t) is sent straight out, without any further transformation. So, at each time step, some memories are dropped and some memories are added.
After the addition operation, the long term state is copied and passed through the tanh function, and then the result is filtered by the output gate. This produces the short-term state h(t) (which is equal to the cell’s output for this time step y(t)).
The 3 gates perform the following functions:
•	The forget gate (controlled by f(t)) controls which parts of the long-term state should be erased. 
•	The input gate (controlled by i(t)) controls which parts of g(t) should be added to the long-term state (this is why we said it was only “partially stored”). 
•	The output gate (controlled by (t)) controls which parts of the long-term state should be read and output at this time step (both to h(t)) and y(t).
Forward Pass: Unrolled Network
The unrolled network during the forward pass is shown below. The gates have not been shown for brevity. You can see that the cell state at time T, cT is responsible for computing hT as well as the next cell state cT+1. At each time step, the cell output hT is shown to be passed to some more layers on which a cost function CT is computed, as the way an LSTM would be used in a typical application like captioning or language modeling.

 
ht= ot⊙tanh(ct)
ct=it⊙at+ft⊙ct−1
zt =W×It

Backward Pass: Unrolled Network
 
The unrolled network during the backward pass is shown above. All the arrows in the previous image have now changed their direction. The cell state at time T, cT receives gradients from hT as well as the next cell state cT +1. At any time step T, these two gradients are accumulated before being backpropagated to the layers below the cell and the previous time steps.
Every gate in a circuit diagram gets some inputs and can right away compute two things: 1. its output value and 2. the local gradient of its inputs with respect to its output value.
once the forward pass is over, during backpropagation the gate will eventually learn about the gradient of its output value on the final output of the entire circuit. Chain rule says that the gate should take that gradient and multiply it into every gradient it normally computes for all of its inputs.
Backpropagation can be thought of as gates communicating to each other (through the gradient signal) whether they want their outputs to increase or decrease (and how strongly), so as to make the final output value higher.
Backward Pass: Output
 
Since we have obtained the value of ht  from the forward pass hence for Error E, δht =∂E/∂ht. 
Now for Backpropagation, we need to find the derivative δot,δct 
i.	∂E/ ∂ot
 
ii.	∂E/ ∂ct
 
Backward Pass: LSTM Memory Cell Update
 
From Forward pass, ct=it⊙at+ft⊙ct−1
We know that δct=∂E/∂ct hence, we need to find δit, δat, δft, δct−1
 

Backward Pass: Input and Gate Computation
 
From Forward pass, zt =W×It
Hence we need to find δzt ,δWt
 
 
If input x has T time-steps, i.e. x=[x1,x2,⋯,xT], then
 
W  is then updated using an appropriate Stochastic Gradient Descent solver.
We try to break up our function into modules for which you can easily derive local gradients, and then chain them with chain rule. Crucially, we almost never want to write out these expressions on paper and differentiate them symbolically in full, because we never need an explicit mathematical equation for the gradient of the input variables. Hence, decompose our expressions into stages such that we can differentiate every stage independently (the stages will be matrix vector multiplies, or max operations, or sum operations, etc.) and then backprop through the variables one step at a time.


Conclusion:

We saw how backpropagation works in different types of Neural network. Although the mathematical equations for all networks are different but the fundamental concept of backpropagation is same for all networks. Make a forward pass, calculate the Loss, make a backward pass to find derivatives and update the weights so as to have the loss function minimized. 
Also, backpropagation works if and only if the activation functions used are differentiable. We should select the activation function whose derivative is faster to compute. Faster the speed of computing derivative, faster is backpropagation. Also, keeping all the datapoints in RAM and computing derivatives is not memory efficient. Hence, minibatch based backpropagation is used. 

References:
http://cs231n.github.io/optimization-2/
 https://www.edureka.co/blog/backpropagation/
http://neuralnetworksanddeeplearning.com/chap2.html
https://medium.com/@jayeshbahire/perceptron-and-backpropagation-970d752f4e44
https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
https://grzegorzgwardys.wordpress.com/2016/04/22/8/
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/

