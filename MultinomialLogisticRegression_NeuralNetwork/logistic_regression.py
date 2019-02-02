from utils import *

# *CODE FOR PART 3.1 IN THIS CELL*

class MultinomialLogisticRegressionClassifier:
    def __init__(self, weight_init_sd=100.0):
        """
        Initializes model parameters to values drawn from the Normal
        distribution with mean 0 and standard deviation `weight_init_sd`.
        """
        self.weight_init_sd = weight_init_sd

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        dis=torch.distributions.normal.Normal(0.0,self.weight_init_sd)
        self.w=torch.autograd.Variable(dis.sample((784,10)), requires_grad=True)
        self.b=torch.autograd.Variable(dis.sample((1,10)),requires_grad=True)


        


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs):
        """
        Performs the forward pass through the model.
        
        Expects `inputs` to be a Tensor of shape (batch_size, 1, 28, 28) containing
        minibatch of MNIST images.
        
        Inputs should be flattened into a Tensor of shape (batch_size, 784),
        before being fed into the model.
        
        Should return a Tensor of logits of shape (batch_size, 10).
        """
        
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        batch_size=inputs.shape[0]
        re = torch.matmul(inputs.view(batch_size,784),self.w)+self.b

        return torch.softmax(re,dim=1)

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

        return [self.w,self.b]
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
            
        return (torch.abs(torch_params).sum())
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
            
        return (torch.sqrt(torch.pow(torch_params, 2).sum()))
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
# *CODE FOR PART 3.2 IN THIS CELL - EXAMPLE WITH DEFAULT PARAMETERS PROVIDED *
model = MultinomialLogisticRegressionClassifier(weight_init_sd=1e-3)
res = run_experiment(
    model,
    optimizer=optim.Adamax(model.parameters(), 0.01),
    train_loader=train_loader_0,
    val_loader=val_loader_0,
    test_loader=test_loader_0,
    n_epochs=10,
    l1_penalty_coef=0.0,
    l2_penalty_coef=0.0,
    suppress_output=False
)