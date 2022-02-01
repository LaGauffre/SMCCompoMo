#!/bin/bash
# Run all the notebooks which we need to be timed on the AWS server for the computational runtimes table 
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute well_spec_500.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute well_spec_1000.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute well_spec_2000.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute miss_spec_500.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute miss_spec_1000.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute miss_spec_2000.ipynb



