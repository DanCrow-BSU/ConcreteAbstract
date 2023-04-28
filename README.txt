The code for this project can be downloaded from:
https://drive.google.com/drive/folders/1hUAY7ze6NliOS00dlxbTLg8V4JiH_XOR?usp=share_link

Note: There is a copy of this file in the google drive folder.


Getting started:

A conda environment can be created with all the necessary packages by
running the following commands:

conda create -y --name conabs
conda activate conabs
conda install -y -c anaconda jupyter pandas numpy scikit-learn
conda install -y -c conda-forge nltk matplotlib tqdm
pip install svgling


Once the environment is up and running run the following notebook:
evaluate.ipynb

This will demonstrate the usage of the ConcreteAbstract class, which
contains the main code for this project.
The code for this class is in the file:
concreteabstract.py

Technically these are the only two files you may care to look at, but it may 
be worth looking at the documentation while reviewing the code:
documentation.ipynb

There are some other files, but they can be safely ignored, unless you really
want to get into things.  All files are listed below along with a short
description.




Files:
evaluate.ipynb
 - Start here for a quick demonstration of the ConcreteAbstract class.

concreteabstract.py
 - Contains the code for the ConcreteAbstract class.
 - This is where all the real code is.

documentation.ipynb
 - Documents the code in concreteabstract.py.
 - Read this in conjunction with reading the code.

concrete-abstract.ipynb
 - Mostly the same code in the ConcreteAbstract class, but broken down into 
   cells to make testing parts of the code easier.
 - This file may be useful for understanding the code.

explore.ipynb
 - Code where I explore different ideas/tests.
 - You can safely ignore this file.

ddata/AC_ratings_google3m_koeper_SiW.csv
 - The Abstract/Concrete word ratings
 - https://aclanthology.org/L16-1413/
 - Köper, Maximilian, and Sabine Schulte im Walde. 2016. “Automatically Generated Affective Norms of Abstractness, Arousal, Imageability and Valence for 350 000 GErman Lemmas.” In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), 2595–98. Portorož, Slovenia: European Language Resources Association (ELRA).

ddata/clip.bertvocab.embeddings.513.txt
 - WAC embeddings file
 - Kennington, Casey, and Osama Natouf. 2022. “The Symbol Grounding Problem Re-Framed as Concreteness-Abstractness Learned through Spoken Interaction.” In Proceedings of the 26th Workshop on the Semantics and Pragmatics of Dialogue - Full Papers. http://semdial.org/anthology/papers/Z/Z22/Z22-3010/.

README
 - This file!



Some of the code and coding ideas in this project came from:
Kennington, Casey, and Osama Natouf. 2022. “The Symbol Grounding Problem Re-Framed as Concreteness-Abstractness Learned through Spoken Interaction.” In Proceedings of the 26th Workshop on the Semantics and Pragmatics of Dialogue - Full Papers. http://semdial.org/anthology/papers/Z/Z22/Z22-3010/.
