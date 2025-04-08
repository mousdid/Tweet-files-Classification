from cs585_P03_A20596524_EXTRA import loader, Classifier
import sys

def main():
    #to get the arguments
    if len(sys.argv) == 3:
        try:
            algo = int(sys.argv[1])
            train_size = int(sys.argv[2])
        except ValueError:
            algo = 0
            train_size = 80
        
        if not (50 <= train_size <= 80) or (algo not in [0, 1]):
            train_size = 80
            algo = 0
    else:
        train_size = 80
        algo = 0
    #
    TRAIN_SIZE = train_size / 100.0  
    path = 'Dataset/'
    ld = loader(path)
    merged_data = ld.load_data()
    ld.export_csv(merged_data, "merged_output.csv")

    ld.preprocess_data()  

if __name__ == "__main__":
    main()





