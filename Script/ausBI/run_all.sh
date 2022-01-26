#!/bin/bash
# Run all the notebooks which we need to be timed on the AWS server for the computational runtimes table 
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute ausBI_single_models.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute ausBI_spliced_models.ipynb



