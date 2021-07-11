# Computation-Intelligence-Projects
These are the two projects I did for my Computation Intelligence class
### Project 1 
In the first project, we are given the MNIST dataset, and we first want to be able to cluster these numbers using the K-Means algorithm, in order to have a number classifier which has a high accuracy on the test set

For the second part of the project, we try the classification with a standard MLP, and then we compare it with the previous results we obtained

Finally, we are looking for a way to use neural networks in order to perform unsupervised clustering. After some research, I found that the most efficient neural network type in order to do this task. is the [self-organizing map](https://en.wikipedia.org/wiki/Self-organizing_map) (AKA Kohonen map) 

### Project 2 
In the second project, we are given a movie dataset which contains two columns, the first column contains the genres a movie belongs to, and the second column is the overview of the movies.\
We are Tasked with building a system that can predict the genres a movie belongs to, based on the overview of it.

In order to do this, we first use one of the two NLP algorithms [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) or [Tf-Idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) in order to extract features for each of the overviews, and then we build a classifier which uses the weights we obtained, as inputs. 

Finally, we look for an efficient way to mix both of the algorithms in order to create a classifier which gives us better results  

* Note that in order to find the feature space of overviews using TF-IDF, we first find the weight vector for the words in our vocabulary, and then for each of the overviews, we find the average weight of the words it contains, and finally we use what is obtained as the feature space for the overview
