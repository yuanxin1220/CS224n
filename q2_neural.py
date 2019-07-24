import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid,sigmoid_grad
from q2_gradcheck import gradcheck_naive
Count=0
def forward_backword_prop(data,labels,params,dimentions):
    ### Unpack network parameters (do not modify)
    ofs=0
    Dx,H,Dy=(dimentions[0],dimentions[1],dimentions[2])

    W1=np.reshape(params[ofs:ofs+Dx*H],(Dx,H))
    ofs+=Dx*H
    b1=np.reshape(params[ofs:ofs+H],(1,H))
    ofs+=H
    W2=np.reshape(params[ofs:ofs+H*Dy],(H,Dy))
    ofs+=H*Dy
    b2=np.reshape(params[ofs:ofs+Dy],(1,Dy))

    ### YOUR CODE HERE: forward propagation
    h=sigmoid(np.dot(data,W1)+b1) # (1,H)
    yhat=softmax(np.dot(h,W2)+b2) # (1,Dy)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    cost=np.sum(-np.log(yhat[labels==1]))/data.shape[0]

    d3=(yhat-labels)/data.shape[0] # (1,Dy)
    gradW2=np.dot(h.T,d3) # (H,Dy)
    gradb2=np.sum(d3,0,keepdims=True) # (1,Dy)

    dh=np.dot(d3,W2.T) # (1,H)
    grad_h=sigmoid_grad(h)*dh # (1,H)

    gradW1=np.dot(data.T,grad_h)
    gradb1=np.sum(grad_h,0)
    ### END YOUR CODE

    ### Stack gradients (do not modify
    grad=np.concatenate((gradW1.flatten(),gradb1.flatten(),
                         gradW2.flatten(),gradb2.flatten()))
    return cost,grad

def sanity_check():
    print("Running sanity check...")

    N=20
    dimentions=[10,5,10]
    data=np.random.randn(N,dimentions[0]) # each row will be a datum
    labels=np.zeros((N,dimentions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimentions[2]-1)]=1

    params=np.random.randn((dimentions[0]+1)*dimentions[1]+
                           (dimentions[1]+1)*dimentions[2],)# 11*5+6*10=115

    gradcheck_naive(lambda params:
                    forward_backword_prop(data,labels,params,dimentions),params)

def your_sanity_checks():
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == '__main__':
    sanity_check()
    your_sanity_checks()