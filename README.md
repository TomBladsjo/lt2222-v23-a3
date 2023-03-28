# LT2222 V23 Assignment 3


## Part 1: a3_features.py

This script takes three positional arguments (inputdir, outputfile and dims) and one optional argument (--test or -T for testsize). "inputdir" is the path to the directory where the data is stored, in this case "/scratch/lt2222-v23/enron_sample/". "outputfile" will be the name of the file written out containing the table of instances. This will be in csv-format. "dims" is the dimension of the representation of the document, i.e. the number of features, given as an integer. Testsize, also an integer, is the percentage of instances to be labelled as "test" (the default is 20 if nothing is provided).

## Part 2 & 3: a3_model.py & augmenting the model

a3_model.py takes one positional argument (featurefile) and three optional arguments (--hidden/-H, --nonlinearity/-nl and --epochs/-E). "featurefile" is the name of the file containing the table of instances and features. 

In its simplest form, without any optional arguments, this script takes the data in the provided file, divides it into training and testing set and trains a simple, linear model with no hidden layers for ten epochs. When done training, the model is tested on the testset and a confusion matrix of the result is printed out. 

"--hidden", if provided without any additional information, will add a hidden layer of default dimension (the mean of the input and output dimensions) to the model. Otherwise, the desired dimension of the layer can be specified as an integer.

"--nonlinearity" adds a nonlinear activation function beween the layers of the model. The type of function must be provided. The options are "tanh" (hyperbolic tangent) and "relu" (rectified linear unit).

"--epochs", if provided, specifies the number of training epochs. The default is 10. 


### Testing some different settings
I tried out the options for non-linearity in combination with different hidden layer sizes. The model was trained for 10 epochs with each setting. The resulting accuracies on the test set can be seen below.

    layer size non-linearity  accuracy
           10        False  0.646154
           10         relu  0.632479
           10         tanh  0.685470
           50        False  0.656410
           50         relu  0.687179
           50         tanh  0.694017
          100        False  0.601709
          100         relu  0.664957
          100         tanh  0.664957
          500        False  0.193162
         500         relu  0.606838
         500         tanh  0.557265
        1000        False  0.317949
        1000         relu  0.548718
        1000         tanh  0.319658

It seems that a smaller hidden layer generally works better than a big one, and that adding Relu or Tanh as an activation function usually (but not always!) improves the performance. Tanh seems to work better than Relu on smaller hidden layer sizes, but worse on bigger sizes. The best result of this experiment came from a hidden layer size 50 and Tanh as non-linearity. 



## Part 4: Enron data ethics


Just as it says in the assignment, this is a complex questions with no clear answers (at least not to someone like me who does not know a lot about law). 

From what I know about research and research ethics in general, it seems like normally when you perform any sort of experiment involving human participants and their data, there are strict rules you need to follow about the ways in which the data are gathered, handled and distributed. For example, you need to obtain informed consent - that is, you need to inform the participants on how their data is going to be used, at least in broad strokes. The data is also generally anonymised before use. 

On the other hand, the emails in the Enron corpus were the legal property of the company and, as I understand it, became public documents when they were used as evidence for criminal charges. I have not managad to find any written discussion on the mismatch in data handling in research settings when dealing with public documents as opposed to consensually obtained data, but it seems to me that this might be an area where the laws are not really up to date with technology and the ways large amounts of data can be shared and used nowadays, and the consequences that can have. For example, when looking through the data, I noticed that a lot of the content did not seem to be business- or company-related. I saw several personal names, phone numbers and addresses that could possibly be to personal homes. Some of the names also seemed to belong to people outside of the company, such as employees' friends or family members - people who were not targets of the investigation and who presumably did not consent to having their personal information made public. 

So, in conclusion, while the question of data ownership might be simple from a legal point of view (as it more or less seems to be from what I read), it is definitely not unproblematic from an ethical point of view. At the very least, I think it would have been a good idea to anonymise the data before releasing it. But maybe that was not a common practice at the time?


## Bonus A: Plotting

"a3_model_plotting.py" uses the model in "a3_model.py", trains and tests it with some different hidden layer sizes and writes the results to a plot, saved as a png-file.

It takes two positional arguments(featurefile and outfile) and five optional arguments (--start, --stop, --step, --nonlinearity and --epochs).

"featurefile" is the name of the file containing the table of instances and features.

"outfile" will be the name of the resulting png-file.

"--start" is the smallest hidden layer size to test. The default value is 5.

"--stop" is the largest hidden layer size to test. The default value is 200.

"--step" is the step size when testing different hidden layer sizes. The default value is 10.

"--nonlinearity" is the type of nonlinear activation function to be used. The options are 'tanh' or 'relu'."

"--epochs" is the number of epochs to train the model in each setting. The default value is 10.











