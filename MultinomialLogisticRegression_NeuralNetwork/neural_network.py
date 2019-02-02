from utils import *

# *CODE FOR PART 4.1 IN THIS CELL*

class MultilayerClassifier:
    def __init__(self, activation_fun="sigmoid", weight_init_sd=1.0):
        """
        Initializes model parameters to values drawn from the Normal
        distribution with mean 0 and standard deviation `weight_init_sd`.
        """
        super().__init__()
        self.activation_fun = activation_fun
        self.weight_init_sd = weight_init_sd

        if self.activation_fun == "relu":
            self.activation = F.relu
        elif self.activation_fun == "sigmoid":
            self.activation = torch.sigmoid
        elif self.activation_fun == "tanh":
            self.activation = torch.tanh
        else:
            raise NotImplementedError()
        
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        dis=torch.distributions.normal.Normal(0.0,self.weight_init_sd)
        self.w1=torch.autograd.Variable(dis.sample((784,128)), requires_grad=True)
        self.b1=torch.autograd.Variable(dis.sample((1,128)),requires_grad=True)
        self.w2=torch.autograd.Variable(dis.sample((128,64)), requires_grad=True)
        self.b2=torch.autograd.Variable(dis.sample((1,64)),requires_grad=True)
        self.w3=torch.autograd.Variable(dis.sample((64,32)), requires_grad=True)
        self.b3=torch.autograd.Variable(dis.sample((1,32)),requires_grad=True)
        self.w4=torch.autograd.Variable(dis.sample((32,10)), requires_grad=True)
        self.b4=torch.autograd.Variable(dis.sample((1,10)),requires_grad=True)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs):
        """
        Performs the forward pass through the model.
        
        Expects `inputs` to be Tensor of shape (batch_size, 1, 28, 28) containing
        minibatch of MNIST images.
        
        Inputs should be flattened into a Tensor of shape (batch_size, 784),
        before being fed into the model.
        
        Should return a Tensor of logits of shape (batch_size, 10).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        batch_size=inputs.shape[0]
        x1 = self.activation(torch.matmul(inputs.view(batch_size,784),self.w1)+self.b1)
        x2 = self.activation(torch.matmul(x1,self.w2)+self.b2)
        x3 = self.activation(torch.matmul(x2,self.w3)+self.b3)
        x4 = torch.softmax(torch.matmul(x3,self.w4)+self.b4, dim=1)
        return x4
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def parameters(self):
        """
        Should return an iterable of all the model parameter Tensors.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return [self.w1, self.b1,self.w2,self.b2,self.w3,self.b3,self.w4,self.b4]
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        
    
    def l1_weight_penalty(self):
        """
        Computes and returns the L1 norm of the model's weight vector (i.e. sum
        of absolute values of all model parameters).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        params_list = []
        for param in self.parameters():
            params_list.append(param.view(-1))
        torch_params = torch.cat(params_list)
        l1=(torch.abs(torch_params).sum())
#         print('l1: ',l1)
            
        return l1
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def l2_weight_penalty(self):
        """
        Computes and returns the L2 weight penalty (i.e. 
        sum of squared values of all model parameters).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        params_list = []    
        for param in self.parameters():
            params_list.append(param.view(-1))
        torch_params = torch.cat(params_list)
        l2=(torch.sqrt(torch.pow(torch_params, 2).sum()))
#         print('l2: ', l2)
        return l2
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################



# *CODE FOR PART 4.2 IN THIS CELL - EXAMPLE WITH DEFAULT PARAMETERS PROVIDED *

model = MultilayerClassifier(activation_fun='relu', weight_init_sd=1e-6)
res = run_experiment(
    model,
    optimizer=optim.Adamax(model.parameters(),5e-3),
    train_loader=train_loader_0,
    val_loader=val_loader_0,
    test_loader=test_loader_0,
    n_epochs=10,
    l1_penalty_coef=1e-5,
    l2_penalty_coef=1e-5,
    suppress_output=False
)