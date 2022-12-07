# On the Personalization and Robustness of Gossip Learning 


## Personalization in Gossip Learning 

The branches TensorFlow & Pytorch are our implementation of the paper:

Yacine Belal, Aurélien Bellet, Sonia Ben Mokhtar and Vlad Nitu (2022). [PEPPER: Empowering User-Centric Recommender Systems over Gossip Learning.](https://dl.acm.org/doi/10.1145/3550302)
in Proceedings of the 2022 ACM International Joint Conference on Pervasive and Ubiquitous Computing and Proceedings 
of the 2022 ACM International Symposium on Wearable Computers (UbiComp ‘22). This paper features our interest in designing model 
aggregation functions that improve the "tail users" recommendation quality in a fully decentralized learning setting.

### Abstract

> Recommender systems are proving to be an invaluable tool for extracting user-relevant content helping users in their daily activities 
(e.g., finding relevant places to visit, content to consume, items to purchase). However, to be effective, these systems need to collect 
and analyze large volumes of personal data (e.g., location check-ins, movie ratings, click rates .. etc.), which exposes users to numerous 
privacy threats. In this context, recommender systems based on Federated Learning (FL) appear to be a promising solution for enforcing privacy 
as they compute accurate recommendations while keeping personal data on the users' devices. However, FL, and therefore FL-based recommender systems, 
rely on a central server that can experience scalability issues besides being vulnerable to attacks. To remedy this, we propose PEPPER, a 
decentralized recommender system based on gossip learning principles. In PEPPER, users gossip model updates and aggregate them asynchronously. 
At the heart of PEPPER reside two key components: a personalized peer-sampling protocol that keeps in the neighborhood of each node, a proportion 
of nodes that have similar interests to the former and a simple yet effective model aggregation function that builds a model that is better suited 
to each user. Through experiments on three real datasets implementing two use cases: a location check-in recommendation and a movie recommendation, 
we demonstrate that our solution converges up to 42% faster than with other decentralized solutions providing up to 9% improvement on average 
performance metric such as hit ratio and up to 21% improvement on long tail performance compared to decentralized competitors.



## Resilience of Gossip Learning to Privacy Attacks
- The branch DUMIA (Decentralized Unsupervised Membership Inference Attack) is an embryonic idea that comes from the observation that 
the items' embeddings of recommendation models form two clear clusters. This can be used to infer liked/interacted with items of a particular user
through her model.
- The branch TensorFlow-Attacked is our implementation of an attack through model evaluation that we have observed to be more impacting on 
centralized architectures.

## Robustness of GL to Poisoning Attacks
FedAtt-Implementation contains the code for the [FedAttack: Effective and Covert Poisoning Attack on Federated Recommendation via Hard Sampling](https://dl.acm.org/doi/abs/10.1145/3534678.3539119)
which tackled none-detectable poisoning attacks on federated recommendation models.

## Software implementation
The results of this paper were produced through extensive simulations which were conducted using [OMNetPy](https://github.com/mmodenesi/omnetpy),  
a python interfacing of [OMNet++](https://omnetpp.org/), a network simulation platform that makes it possible to design architectures with a certain
flexibility, to give the nodes an edge-device like behavior. Associated with Keras/Theano, it allowed us to train models at the level of each node and to
exchange these models according to highly customizable communication protocols.



## Dependencies
You'll need a working Python environment to run the code. As well as the following list of packages and dependencies :
locales 
wget 
build-essential 
gcc 
g++ 
bison 
flex 
perl 
qt5-default 
tcl-dev 
tk-dev 
libxml2-dev 
zlib1g-dev 
default-jre 
doxygen 
graphviz 
openmpi-bin 
libopenmpi-dev 
python3.8 
libqt5opengl5-dev 
libpcap-dev nemiver 
pip
pybind11==2.4.3
keras==1.0.7
theano==0.8.0
numpy 
matplotlib 
wandb

You'll also need to install omnet++ 5.6.1 & omnetpy.

## License
All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. 
