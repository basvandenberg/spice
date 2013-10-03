
spice
=====

The spice package can be used for calculating sequence-based protein features,
visualizing the obtained features, and training and testing of protein
classifiers using these features.

This featext.py module can be used for sequence-based protein feature
extraction.  it uses the featmat.py module to manage the labeled feature
matrix, an m x n matrix for m proteins and n features, and the dataset.py
module to manage the set of proteins and their corresponding labels.

The classification.py module is a layer on top of scikit-learn that can be used
to construct protein classifiers and the classify.py module can be used to test
new protein sequences on an allready trained classifier.

The project\_management.py module is used by the SPiCE website to manage user
projects.


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


Usage
=====

The software can be using the import statement, for example:

    from spice import featext

Four command-line tools are also provided:

- featext
- classification
- classify

