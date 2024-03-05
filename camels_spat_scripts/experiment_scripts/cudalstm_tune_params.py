## Import necessary libraries
import os
import yaml

## Functions

def main():
    
    # Load the parameters to be tuned as a dictionary
    with open('cudalstm_params.yml', 'r') as ymlfile:
        cudalstm_params = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
    print(cudalstm_params)
    
    


if __name__ == '__main__':
    
    main()