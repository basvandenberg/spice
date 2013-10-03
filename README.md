
spice
=====

The spice package can be used for calculating sequence-based protein features,
visualizing the obtained features, and training and testing of protein
classifiers using these features.


featext.py
----------
This module can be used for sequence-based protein feature extraction.


featmat.py
----------
This module manages the labeled feature matrix. An m x n matrix for m proteins
and n features that can be build using the featext module.


dataset.py
----------
This module manages the set of proteins and their corresponding labels.


classification.py
-----------------

A layer on top of scikit-learn that can be used to construct protein
classifiers.


classify.py
-----------

This module can be used to test new protein sequences on an allready trained
classifier.


project\_management.py
---------------------

This module is used by the SPiCE website to manage user projects.


Dependencies
============

The following software is required to run spice:

- numpy >= 1.7.1
- scipy >= 0.12.0
- matplotlib >= 1.2.2
- scikit-learn >= 0.14.1

The biopy package that can also be found on my github repository is also
required:

- biopy >= 0.1.0"


Install
=======

On linux systems, the sofware can be installed using:

    sudo python setup.py install
